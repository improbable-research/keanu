package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.LogProbGradient;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Hamiltonian Monte Carlo is a method for obtaining samples from a probability
 * distribution with the introduction of a momentum variable.
 * <p>
 * Algorithm 1: "Hamiltonian Monte Carlo".
 * The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo
 * https://arxiv.org/pdf/1111.4246.pdf
 */
public class Hamiltonian {

    private Hamiltonian() {
    }

    public static NetworkSamples getPosteriorSamples(final BayesianNetwork bayesNet,
                                                     final List<? extends Vertex> fromVertices,
                                                     final int sampleCount,
                                                     final int leapFrogCount,
                                                     final double stepSize) {

        return getPosteriorSamples(bayesNet, fromVertices, sampleCount, leapFrogCount, stepSize, new KeanuRandom());
    }

    public static NetworkSamples getPosteriorSamples(final BayesianNetwork bayesNet,
                                                     final List<? extends Vertex> fromVertices,
                                                     final int sampleCount,
                                                     final int leapFrogCount,
                                                     final double stepSize,
                                                     final KeanuRandom random) {

        bayesNet.cascadeObservations();

        final List<Vertex<DoubleTensor>> latentVertices = bayesNet.getContinuousLatentVertices();
        final List<Vertex> probabilisticVertices = bayesNet.getLatentAndObservedVertices();

        final Map<Long, List<?>> samples = new HashMap<>();
        addSampleFromVertices(samples, fromVertices);

        Map<Long, DoubleTensor> position = new HashMap<>();
        cachePosition(latentVertices, position);
        Map<Long, DoubleTensor> positionBeforeLeapfrog = new HashMap<>();

        Map<Long, DoubleTensor> gradient = LogProbGradient.getJointLogProbGradientWrtLatents(
            bayesNet.getLatentAndObservedVertices()
        );
        Map<Long, DoubleTensor> gradientBeforeLeapfrog = new HashMap<>();

        final Map<Long, DoubleTensor> momentum = new HashMap<>();
        final Map<Long, DoubleTensor> momentumBeforeLeapfrog = new HashMap<>();

        double logOfMasterPBeforeLeapfrog = bayesNet.getLogOfMasterP();

        final Map<Long, ?> sampleBeforeLeapfrog = new HashMap<>();

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
                    probabilisticVertices
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
                Map<Long, DoubleTensor> tempSwap = position;
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
        }

        return new NetworkSamples(samples, sampleCount);
    }

    private static void cachePosition(List<Vertex<DoubleTensor>> latentVertices, Map<Long, DoubleTensor> position) {
        for (Vertex<DoubleTensor> vertex : latentVertices) {
            position.put(vertex.getId(), vertex.getValue());
        }
    }

    private static void initializeMomentumForEachVertex(List<Vertex<DoubleTensor>> vertexes,
                                                        Map<Long, DoubleTensor> momentums,
                                                        KeanuRandom random) {
        for (Vertex<DoubleTensor> currentVertex : vertexes) {
            momentums.put(currentVertex.getId(), random.nextGaussian(currentVertex.getShape()));
        }
    }

    private static void cache(Map<Long, DoubleTensor> from, Map<Long, DoubleTensor> to) {
        for (Map.Entry<Long, DoubleTensor> entry : from.entrySet()) {
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
     * @param gradient                 gradient at current position
     * @param momentums                current vertex momentums
     * @param stepSize
     * @param probabilisticVertices    all vertices that impact the joint posterior (masterP)
     * @return the gradient at the updated position
     */
    private static Map<Long, DoubleTensor> leapfrog(final List<Vertex<DoubleTensor>> latentVertices,
                                                    final Map<Long, DoubleTensor> position,
                                                    final Map<Long, DoubleTensor> gradient,
                                                    final Map<Long, DoubleTensor> momentums,
                                                    final double stepSize,
                                                    final List<? extends Vertex> probabilisticVertices) {

        final double halfTimeStep = stepSize / 2.0;

        Map<Long, DoubleTensor> momentumsAtHalfTimeStep = new HashMap<>();

        //Set `r = r + (eps/2)dTL(T)
        for (Map.Entry<Long, DoubleTensor> vertexMomentum : momentums.entrySet()) {
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
        Map<Long, DoubleTensor> newGradient = LogProbGradient.getJointLogProbGradientWrtLatents(
            probabilisticVertices
        );

        for (Map.Entry<Long, DoubleTensor> halfTimeStepMomentum : momentumsAtHalfTimeStep.entrySet()) {
            final DoubleTensor updatedMomentum = newGradient.get(halfTimeStepMomentum.getKey()).times(halfTimeStep).plusInPlace(halfTimeStepMomentum.getValue());
            momentums.put(halfTimeStepMomentum.getKey(), updatedMomentum);
        }

        return newGradient;
    }

    private static double getLikelihoodOfLeapfrog(final double logOfMasterPAfterLeapfrog,
                                                  final double previousLogOfMasterP,
                                                  final Map<Long, DoubleTensor> leapfroggedMomentum,
                                                  final Map<Long, DoubleTensor> momentumPreviousTimeStep) {

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

    private static double dotProduct(Map<Long, DoubleTensor> momentums) {
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
    private static void takeSample(Map<Long, ?> sample, List<? extends Vertex> fromVertices) {
        for (Vertex<?> vertex : fromVertices) {
            putValue(vertex, sample);
        }
    }

    private static <T> void putValue(Vertex<T> vertex, Map<Long, ?> target) {
        ((Map<Long, T>) target).put(vertex.getId(), vertex.getValue());
    }

    /**
     * This is used when a leapfrog is rejected. At that point the vertices are in a post
     * leapfrog state and a pre-leapfrog sample must be used.
     *
     * @param samples
     * @param cachedSample a cached sample from before leapfrog
     */
    private static void addSampleFromCache(Map<Long, List<?>> samples, Map<Long, ?> cachedSample) {
        for (Map.Entry<Long, ?> sampleEntry : cachedSample.entrySet()) {
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
    private static void addSampleFromVertices(Map<Long, List<?>> samples, List<? extends Vertex> fromVertices) {
        for (Vertex<?> vertex : fromVertices) {
            addSampleForVertex(vertex.getId(), vertex.getValue(), samples);
        }
    }

    private static <T> void addSampleForVertex(long id, T value, Map<Long, List<?>> samples) {
        List<T> samplesForVertex = (List<T>) samples.computeIfAbsent(id, v -> new ArrayList<T>());
        samplesForVertex.add(value);
    }

}
