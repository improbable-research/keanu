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

        return getPosteriorSamples(bayesNet, fromVertices, sampleCount, leapFrogCount, stepSize, Vertex.getDefaultRandom());
    }

    public static NetworkSamples getPosteriorSamples(final BayesNet bayesNet,
                                                     final List<? extends Vertex> fromVertices,
                                                     final int sampleCount,
                                                     final int leapFrogCount,
                                                     final double stepSize,
                                                     final Random random) {

        final List<Vertex<Double>> latentVertices = bayesNet.getContinuousLatentVertices();
        final Map<Long, Long> latentSetAndCascadeCache = VertexValuePropagation.exploreSetting(latentVertices);
        final List<Vertex> probabilisticVertices = bayesNet.getLatentAndObservedVertices();

        final Map<Long, List<?>> samples = new HashMap<>();
        addSampleFromVertices(samples, fromVertices);

        Map<Long, Double> position = new HashMap<>();
        cachePosition(latentVertices, position);
        Map<Long, Double> positionBeforeLeapfrog = new HashMap<>();

        Map<Long, Double> gradient = LogProbGradient.getJointLogProbGradientWrtLatents(
                bayesNet.getLatentAndObservedVertices()
        );
        Map<Long, Double> gradientBeforeLeapfrog = new HashMap<>();

        final Map<Long, Double> momentum = new HashMap<>();
        final Map<Long, Double> momentumBeforeLeapfrog = new HashMap<>();

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
                Map<Long, Double> tempSwap = position;
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

    private static void cachePosition(List<Vertex<Double>> latentVertices, Map<Long, Double> position) {
        for (Vertex<Double> vertex : latentVertices) {
            position.put(vertex.getId(), vertex.getValue());
        }
    }

    private static Map<Long, Double> initializeMomentumForEachVertex(List<Vertex<Double>> vertexes,
                                                                       Map<Long, Double> momentums,
                                                                       Random random) {
        for (int i = 0; i < vertexes.size(); i++) {
            Vertex currentVertex = vertexes.get(i);
            momentums.put(currentVertex.getId(), random.nextGaussian());
        }
        return momentums;
    }

    private static void cache(Map<Long, Double> from, Map<Long, Double> to) {
        for (Map.Entry<Long, Double> entry : from.entrySet()) {
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
     * @param gradient              gradient at current position
     * @param momentums             current vertex momentums
     * @param stepSize
     * @param probabilisticVertices all vertices that impact the joint posterior (masterP)
     * @return the gradient at the updated position
     */
    private static Map<Long, Double> leapfrog(final List<Vertex<Double>> latentVertices,
                                                final Map<Long, Long> latentSetAndCascadeCache,
                                                final Map<Long, Double> position,
                                                final Map<Long, Double> gradient,
                                                final Map<Long, Double> momentums,
                                                final double stepSize,
                                                final List<? extends Vertex> probabilisticVertices) {

        final double halfTimeStep = stepSize / 2.0;

        Map<Long, Double> momentumsAtHalfTimeStep = new HashMap<>();

        //Set `r = r + (eps/2)dTL(T)
        for (Map.Entry<Long, Double> vertexMomentum : momentums.entrySet()) {
            final double updatedMomentum = vertexMomentum.getValue() + halfTimeStep * gradient.get(vertexMomentum.getKey());
            momentumsAtHalfTimeStep.put(vertexMomentum.getKey(), updatedMomentum);
        }

        //Set `T = T + `r.
        for (Vertex<Double> latent : latentVertices) {
            final double nextPosition = position.get(latent.getId()) + halfTimeStep * momentumsAtHalfTimeStep.get(latent.getId());
            position.put(latent.getId(), nextPosition);
            latent.setValue(nextPosition);
        }

        VertexValuePropagation.cascadeUpdate(latentVertices, latentSetAndCascadeCache);

        //Set `r = `r + (eps/2)dTL(`T)
        Map<Long, Double> newGradient = LogProbGradient.getJointLogProbGradientWrtLatents(
                probabilisticVertices
        );

        for (Map.Entry<Long, Double> halfTimeStepMomentum : momentumsAtHalfTimeStep.entrySet()) {
            final double updatedMomentum = halfTimeStepMomentum.getValue() + halfTimeStep * newGradient.get(halfTimeStepMomentum.getKey());
            momentums.put(halfTimeStepMomentum.getKey(), updatedMomentum);
        }

        return newGradient;
    }

    private static double getLikelihoodOfLeapfrog(final double logOfMasterPAfterLeapfrog,
                                                  final double previousLogOfMasterP,
                                                  final Map<Long, Double> leapfroggedMomentum,
                                                  final Map<Long, Double> momentumPreviousTimeStep) {

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

    private static double dotProduct(Map<Long, Double> momentums) {
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
