package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.network.BayesNet;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.LogProbGradient;

import java.util.*;

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

    public static NetworkSamples getPosteriorSamples(final BayesNet bayesNet,
                                                     final List<DoubleVertex> fromVertices,
                                                     final int sampleCount,
                                                     final int leapFrogCount,
                                                     final double stepSize) {

        return getPosteriorSamples(bayesNet, fromVertices, sampleCount, leapFrogCount, stepSize, new Random());
    }

    public static NetworkSamples getPosteriorSamples(final BayesNet bayesNet,
                                                     final List<? extends Vertex<?>> fromVertices,
                                                     final int sampleCount,
                                                     final int leapFrogCount,
                                                     final double stepSize,
                                                     final Random random) {

        final List<Vertex<Double>> latentVertices = bayesNet.getContinuousLatentVertices();
        final Map<String, Long> latentSetAndCascadeCache = VertexValuePropagation.exploreSetting(latentVertices);
        final List<Vertex<?>> probabilisticVertices = bayesNet.getVerticesThatContributeToMasterP();

        final Map<String, List<?>> samples = new HashMap<>();
        addSampleFromVertices(samples, fromVertices);

        Map<String, Double> position = new HashMap<>();
        cachePosition(latentVertices, position);
        Map<String, Double> positionBeforeLeapfrog = new HashMap<>();

        Map<String, Double> gradient = LogProbGradient.getJointLogProbGradientWrtLatents(
                bayesNet.getVerticesThatContributeToMasterP()
        );
        Map<String, Double> gradientBeforeLeapfrog = new HashMap<>();

        final Map<String, Double> momentum = new HashMap<>();
        final Map<String, Double> momentumBeforeLeapfrog = new HashMap<>();

        double logOfMasterPBeforeLeapfrog = bayesNet.getLogOfMasterP();

        final Map<String, ?> sampleBeforeLeapfrog = new HashMap<>();

        for (int sampleNum = 1; sampleNum < sampleCount; sampleNum++) {

            cache(position, positionBeforeLeapfrog);
            cache(gradient, gradientBeforeLeapfrog);

            initializeMomentumForEachVertex(latentVertices, momentum, random);
            cache(momentum, momentumBeforeLeapfrog);

            takeSample(sampleBeforeLeapfrog, fromVertices);

            for (int leapFrogNum = 0; leapFrogNum < leapFrogCount; leapFrogNum++) {
                gradient = leapfrog(
                        latentVertices,
                        latentSetAndCascadeCache,
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
                Map<String, Double> tempSwap = position;
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

    private static void cachePosition(List<Vertex<Double>> latentVertices, Map<String, Double> position) {
        for (Vertex<Double> vertex : latentVertices) {
            position.put(vertex.getId(), vertex.getValue());
        }
    }

    private static Map<String, Double> initializeMomentumForEachVertex(List<Vertex<Double>> vertexes,
                                                                       Map<String, Double> momentums,
                                                                       Random random) {
        for (int i = 0; i < vertexes.size(); i++) {
            Vertex currentVertex = vertexes.get(i);
            momentums.put(currentVertex.getId(), random.nextGaussian());
        }
        return momentums;
    }

    private static void cache(Map<String, Double> from, Map<String, Double> to) {
        for (Map.Entry<String, Double> entry : from.entrySet()) {
            to.put(entry.getKey(), entry.getValue());
        }
    }

    /**
     * function Leapfrog(θ, r, )
     * Set ˜r ← r + (eps/2)∇θL(θ)
     * Set ˜θ ← θ + r˜
     * Set ˜r ← r˜ + (eps/2)∇θL(˜θ)
     * return ˜θ, r˜
     *
     * @param latentVertices
     * @param gradient              gradient at current position
     * @param momentums             current vertex momentums
     * @param stepSize
     * @param probabilisticVertices all vertices that impact the joint posterior (masterP)
     * @return the gradient at the updated position
     */
    private static Map<String, Double> leapfrog(final List<Vertex<Double>> latentVertices,
                                                final Map<String, Long> latentSetAndCascadeCache,
                                                final Map<String, Double> position,
                                                final Map<String, Double> gradient,
                                                final Map<String, Double> momentums,
                                                final double stepSize,
                                                final List<Vertex<?>> probabilisticVertices) {

        final double halfTimeStep = stepSize / 2.0;

        Map<String, Double> momentumsAtHalfTimeStep = new HashMap<>();

        //Set ˜r ← r + (eps/2)∇θL(θ)
        for (Map.Entry<String, Double> vertexMomentum : momentums.entrySet()) {
            final double updatedMomentum = vertexMomentum.getValue() + halfTimeStep * gradient.get(vertexMomentum.getKey());
            momentumsAtHalfTimeStep.put(vertexMomentum.getKey(), updatedMomentum);
        }

        //Set ˜θ ← θ + ˜r.
        for (Vertex<Double> latent : latentVertices) {
            final double nextPosition = position.get(latent.getId()) + halfTimeStep * momentumsAtHalfTimeStep.get(latent.getId());
            position.put(latent.getId(), nextPosition);
            latent.setValue(nextPosition);
        }

        VertexValuePropagation.cascadeUpdate(latentVertices, latentSetAndCascadeCache);

        //Set ˜r ← ˜r + (eps/2)∇θL(˜θ)
        Map<String, Double> newGradient = LogProbGradient.getJointLogProbGradientWrtLatents(
                probabilisticVertices
        );

        for (Map.Entry<String, Double> halfTimeStepMomentum : momentumsAtHalfTimeStep.entrySet()) {
            final double updatedMomentum = halfTimeStepMomentum.getValue() + halfTimeStep * newGradient.get(halfTimeStepMomentum.getKey());
            momentums.put(halfTimeStepMomentum.getKey(), updatedMomentum);
        }

        return newGradient;
    }

    private static double getLikelihoodOfLeapfrog(final double logOfMasterPAfterLeapfrog,
                                                  final double previousLogOfMasterP,
                                                  final Map<String, Double> leapfroggedMomentum,
                                                  final Map<String, Double> momentumPreviousTimeStep) {

        final double leapFroggedMomentumDotProduct = (0.5 * dotProduct(leapfroggedMomentum));
        final double previousMomentumDotProduct = (0.5 * dotProduct(momentumPreviousTimeStep));

        final double leapFroggedLikelihood = logOfMasterPAfterLeapfrog - leapFroggedMomentumDotProduct;
        final double previousLikelihood = previousLogOfMasterP - previousMomentumDotProduct;

        final double logLikelihoodOfLeapFrog = leapFroggedLikelihood - previousLikelihood;
        final double likelihoodOfLeapfrog = Math.exp(logLikelihoodOfLeapFrog);

        return Math.min(likelihoodOfLeapfrog, 1.0);
    }

    private static boolean shouldReject(double likelihood, Random random) {
        return likelihood < random.nextDouble();
    }

    private static double dotProduct(Map<String, Double> momentums) {
        double dotProduct = 0.0;
        for (Double momentum : momentums.values()) {
            dotProduct += momentum * momentum;
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
    private static void takeSample(Map<String, ?> sample, List<? extends Vertex<?>> fromVertices) {
        for (Vertex<?> vertex : fromVertices) {
            putValue(vertex, sample);
        }
    }

    private static <T> void putValue(Vertex<T> vertex, Map<String, ?> target) {
        ((Map<String, T>) target).put(vertex.getId(), vertex.getValue());
    }

    /**
     * This is used when a leapfrog is rejected. At that point the vertices are in a post
     * leapfrog state and a pre-leapfrog sample must be used.
     *
     * @param samples
     * @param cachedSample a cached sample from before leapfrog
     */
    private static void addSampleFromCache(Map<String, List<?>> samples, Map<String, ?> cachedSample) {
        for (Map.Entry<String, ?> sampleEntry : cachedSample.entrySet()) {
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
    private static void addSampleFromVertices(Map<String, List<?>> samples, List<? extends Vertex<?>> fromVertices) {
        for (Vertex<?> vertex : fromVertices) {
            addSampleForVertex(vertex.getId(), vertex.getValue(), samples);
        }
    }

    private static <T> void addSampleForVertex(String id, T value, Map<String, List<?>> samples) {
        List<T> samplesForVertex = (List<T>) samples.computeIfAbsent(id, v -> new ArrayList<T>());
        samplesForVertex.add(value);
    }

}
