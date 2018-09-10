package io.improbable.keanu.algorithms.mcmc;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.PosteriorSamplingAlgorithm;
import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.LogProbGradientCalculator;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;

/**
 * Hamiltonian Monte Carlo is a method for obtaining samples from a probability
 * distribution with the introduction of a momentum variable.
 * <p>
 * Algorithm 1: "Hamiltonian Monte Carlo".
 * The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo
 * https://arxiv.org/pdf/1111.4246.pdf
 */
@Builder
public class Hamiltonian implements PosteriorSamplingAlgorithm {

    private static final double DEFAULT_STEP_SIZE = 0.1;
    private static final int DEFAULT_LEAP_FROG_COUNT = 20;

    public static Hamiltonian withDefaultConfig() {
        return withDefaultConfig(KeanuRandom.getDefaultRandom());
    }

    public static Hamiltonian withDefaultConfig(KeanuRandom random) {
        return Hamiltonian.builder()
            .random(random)
            .build();
    }

    @Getter
    @Setter
    @Builder.Default
    private KeanuRandom random = KeanuRandom.getDefaultRandom();

    @Getter
    @Setter
    @Builder.Default
    //the number of times to leapfrog in each sample
    private int leapFrogCount = DEFAULT_LEAP_FROG_COUNT;

    @Getter
    @Setter
    @Builder.Default
    //the amount of distance to move each leapfrog
    private double stepSize = DEFAULT_STEP_SIZE;

    /**
     * Sample from the posterior of a Bayesian Network using the Hamiltonian Monte Carlo algorithm
     *
     * @param bayesNet     The bayesian network to sample from
     * @param fromVertices the vertices to sample from
     * @param sampleCount  the number of samples to take
     * @return Samples taken with Hamiltonian Monte Carlo
     */
    @Override
    public NetworkSamples getPosteriorSamples(final BayesianNetwork bayesNet,
                                              final List<? extends Vertex> fromVertices,
                                              final int sampleCount) {

        bayesNet.cascadeObservations();

        final List<Vertex<DoubleTensor>> latentVertices = bayesNet.getContinuousLatentVertices();
        final LogProbGradientCalculator logProbGradientCalculator = new LogProbGradientCalculator(bayesNet.getLatentOrObservedVertices(), latentVertices);
        final List<? extends Probabilistic> probabilisticVertices = Probabilistic.keepOnlyProbabilisticVertices(bayesNet.getLatentOrObservedVertices());

        final Map<VertexId, List<?>> samples = new HashMap<>();
        addSampleFromVertices(samples, fromVertices);

        Map<VertexId, DoubleTensor> position = new HashMap<>();
        cachePosition(latentVertices, position);
        Map<VertexId, DoubleTensor> positionBeforeLeapfrog = new HashMap<>();

        Map<VertexId, DoubleTensor> gradient = logProbGradientCalculator.getJointLogProbGradientWrtLatents();
        Map<VertexId, DoubleTensor> gradientBeforeLeapfrog = new HashMap<>();

        final Map<VertexId, DoubleTensor> momentum = new HashMap<>();
        final Map<VertexId, DoubleTensor> momentumBeforeLeapfrog = new HashMap<>();

        double logOfMasterPBeforeLeapfrog = bayesNet.getLogOfMasterP();
        final List<Double> logOfMasterPForEachSample = new ArrayList<>();
        logOfMasterPForEachSample.add(logOfMasterPBeforeLeapfrog);

        final Map<VertexId, ?> sampleBeforeLeapfrog = new HashMap<>();

        for (int sampleNum = 1; sampleNum < sampleCount; sampleNum++) {

            cache(position, positionBeforeLeapfrog);
            cache(gradient, gradientBeforeLeapfrog);

            initializeMomentumForEachVertex(latentVertices, momentum, random);
            cache(momentum, momentumBeforeLeapfrog);

            takeSample(sampleBeforeLeapfrog, fromVertices);

            for (int leapFrogNum = 0; leapFrogNum < leapFrogCount; leapFrogNum++) {
                gradient = leapfrog(
                    latentVertices,
                    position,
                    gradient,
                    momentum,
                    stepSize,
                    logProbGradientCalculator
                );
            }

            final double logOfMasterPAfterLeapfrog = bayesNet.getLogOfMasterP();

            final double likelihoodOfLeapfrog = getLikelihoodOfLeapfrog(
                logOfMasterPAfterLeapfrog,
                logOfMasterPBeforeLeapfrog,
                momentum,
                momentumBeforeLeapfrog
            );

            if (shouldReject(likelihoodOfLeapfrog, random)) {

                //Revert to position and gradient before leapfrog
                Map<VertexId, DoubleTensor> tempSwap = position;
                position = positionBeforeLeapfrog;
                positionBeforeLeapfrog = tempSwap;

                tempSwap = gradient;
                gradient = gradientBeforeLeapfrog;
                gradientBeforeLeapfrog = tempSwap;

                addSampleFromCache(samples, sampleBeforeLeapfrog);
            } else {
                addSampleFromVertices(samples, fromVertices);
                logOfMasterPBeforeLeapfrog = logOfMasterPAfterLeapfrog;
            }
            logOfMasterPForEachSample.add(logOfMasterPBeforeLeapfrog);
        }

        return new NetworkSamples(samples, logOfMasterPForEachSample, sampleCount);
    }

    private static void cachePosition(List<Vertex<DoubleTensor>> latentVertices, Map<VertexId, DoubleTensor> position) {
        for (Vertex<DoubleTensor> vertex : latentVertices) {
            position.put(vertex.getId(), vertex.getValue());
        }
    }

    private static void initializeMomentumForEachVertex(List<Vertex<DoubleTensor>> vertexes,
                                                        Map<VertexId, DoubleTensor> momentums,
                                                        KeanuRandom random) {
        for (Vertex<DoubleTensor> currentVertex : vertexes) {
            momentums.put(currentVertex.getId(), random.nextGaussian(currentVertex.getShape()));
        }
    }

    private static void cache(Map<VertexId, DoubleTensor> from, Map<VertexId, DoubleTensor> to) {
        for (Map.Entry<VertexId, DoubleTensor> entry : from.entrySet()) {
            to.put(entry.getKey(), entry.getValue());
        }
    }

    /**
     * function Leapfrog(T, r)
     * Set `r = r + (eps/2)dTL(T)
     * Set `T = T + r`
     * Set `r = r` + (eps/2)dTL(`T)
     * return `T, r`
     *
     * @param latentVertices
     * @param position
     * @param gradient        gradient at current position
     * @param momentums       current vertex momentums
     * @param stepSize
     * @param logProbGradient calculator of the logProb gradients
     * @return the gradient at the updated position
     */
    private static Map<VertexId, DoubleTensor> leapfrog(final List<Vertex<DoubleTensor>> latentVertices,
                                                        final Map<VertexId, DoubleTensor> position,
                                                        final Map<VertexId, DoubleTensor> gradient,
                                                        final Map<VertexId, DoubleTensor> momentums,
                                                        final double stepSize,
                                                        final LogProbGradientCalculator logProbGradient) {

        final double halfTimeStep = stepSize / 2.0;

        Map<VertexId, DoubleTensor> momentumsAtHalfTimeStep = new HashMap<>();

        //Set `r = r + (eps/2)dTL(T)
        for (Map.Entry<VertexId, DoubleTensor> vertexMomentum : momentums.entrySet()) {
            final DoubleTensor updatedMomentum = gradient.get(vertexMomentum.getKey()).times(halfTimeStep).plusInPlace(vertexMomentum.getValue());
            momentumsAtHalfTimeStep.put(vertexMomentum.getKey(), updatedMomentum);
        }

        //Set `T = T + `r.
        for (Vertex<DoubleTensor> latent : latentVertices) {
            final DoubleTensor nextPosition = momentumsAtHalfTimeStep.get(latent.getId()).times(halfTimeStep).plusInPlace(position.get(latent.getId()));
            position.put(latent.getId(), nextPosition);
            latent.setValue(nextPosition);
        }

        VertexValuePropagation.cascadeUpdate(latentVertices);

        //Set `r = `r + (eps/2)dTL(`T)
        Map<VertexId, DoubleTensor> newGradient = logProbGradient.getJointLogProbGradientWrtLatents();

        for (Map.Entry<VertexId, DoubleTensor> halfTimeStepMomentum : momentumsAtHalfTimeStep.entrySet()) {
            final DoubleTensor updatedMomentum = newGradient.get(halfTimeStepMomentum.getKey()).times(halfTimeStep).plusInPlace(halfTimeStepMomentum.getValue());
            momentums.put(halfTimeStepMomentum.getKey(), updatedMomentum);
        }

        return newGradient;
    }

    private static double getLikelihoodOfLeapfrog(final double logOfMasterPAfterLeapfrog,
                                                  final double previousLogOfMasterP,
                                                  final Map<VertexId, DoubleTensor> leapfroggedMomentum,
                                                  final Map<VertexId, DoubleTensor> momentumPreviousTimeStep) {

        final double leapFroggedMomentumDotProduct = (0.5 * dotProduct(leapfroggedMomentum));
        final double previousMomentumDotProduct = (0.5 * dotProduct(momentumPreviousTimeStep));

        final double leapFroggedLikelihood = logOfMasterPAfterLeapfrog - leapFroggedMomentumDotProduct;
        final double previousLikelihood = previousLogOfMasterP - previousMomentumDotProduct;

        final double logLikelihoodOfLeapFrog = leapFroggedLikelihood - previousLikelihood;
        final double likelihoodOfLeapfrog = Math.exp(logLikelihoodOfLeapFrog);

        return Math.min(likelihoodOfLeapfrog, 1.0);
    }

    private static boolean shouldReject(double likelihood, KeanuRandom random) {
        return likelihood < random.nextDouble();
    }

    private static double dotProduct(Map<VertexId, DoubleTensor> momentums) {
        double dotProduct = 0.0;
        for (DoubleTensor momentum : momentums.values()) {
            dotProduct += momentum.pow(2).sum();
        }
        return dotProduct;
    }

    /**
     * This is meant to be used for caching a pre-leapfrog sample. This sample
     * will be used if the leapfrog is rejected.
     *
     * @param sample
     * @param fromVertices
     */
    private static void takeSample(Map<VertexId, ?> sample, List<? extends Vertex> fromVertices) {
        for (Vertex<?> vertex : fromVertices) {
            putValue(vertex, sample);
        }
    }

    private static <T> void putValue(Vertex<T> vertex, Map<VertexId, ?> target) {
        ((Map<VertexId, T>) target).put(vertex.getId(), vertex.getValue());
    }

    /**
     * This is used when a leapfrog is rejected. At that point the vertices are in a post
     * leapfrog state and a pre-leapfrog sample must be used.
     *
     * @param samples
     * @param cachedSample a cached sample from before leapfrog
     */
    private static void addSampleFromCache(Map<VertexId, List<?>> samples, Map<VertexId, ?> cachedSample) {
        for (Map.Entry<VertexId, ?> sampleEntry : cachedSample.entrySet()) {
            addSampleForVertex(sampleEntry.getKey(), sampleEntry.getValue(), samples);
        }
    }

    /**
     * This is used when a leapfrog is accepted. At that point the vertices are in a
     * post leapfrog state.
     *
     * @param samples
     * @param fromVertices vertices from which to create and save new sample.
     */
    private static void addSampleFromVertices(Map<VertexId, List<?>> samples, List<? extends Vertex> fromVertices) {
        for (Vertex<?> vertex : fromVertices) {
            addSampleForVertex(vertex.getId(), vertex.getValue(), samples);
        }
    }

    private static <T> void addSampleForVertex(VertexId id, T value, Map<VertexId, List<?>> samples) {
        List<T> samplesForVertex = (List<T>) samples.computeIfAbsent(id, v -> new ArrayList<T>());
        samplesForVertex.add(value);
    }
}
