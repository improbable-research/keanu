package io.improbable.keanu.algorithms.sampling;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class RejectionSampler {

    private RejectionSampler() {
    }

    public static double getPosteriorProbability(List<? extends Vertex> latentVertices,
                                                 List<? extends Vertex> observedVertices,
                                                 Supplier<Boolean> isSuccess,
                                                 int sampleCount) {
        return getPosteriorProbability(latentVertices, observedVertices, isSuccess, sampleCount, KeanuRandom.getDefaultRandom());
    }

    public static double getPosteriorProbability(List<? extends Vertex> latentVertices,
                                                 List<? extends Vertex> observedVertices,
                                                 Supplier<Boolean> isSuccess,
                                                 int sampleCount,
                                                 KeanuRandom random) {
        int matchedSampleCount = 0;
        int success = 0;

        while (matchedSampleCount < sampleCount) {
            sampleLatents(latentVertices, random);
            if (matchesObservation(observedVertices)) {
                matchedSampleCount++;
                if (isSuccess.get()) {
                    success++;
                }
            }
        }

        if (matchedSampleCount == 0) {
            throw new IllegalStateException("No samples are matching.");
        } else {
            return success / (double) matchedSampleCount;
        }
    }

    /**
     * Samples from the posterior of a Bayesian Network using Rejection Sampling / Accept-Reject.
     * Rejection Sampling samples from a uniform space and retains the samples in the region
     * under the graph of the network's distribution function.
     *
     * @param bayesNet the bayesian network to sample from
     * @param fromVertices the vertices to sample from
     * @param sampleCount the number of samples to take
     * @return Samples taken with Rejection sampling.
     */
    public static NetworkSamples getPosteriorSamples(BayesianNetwork bayesNet,
                                                     List<Vertex<?>> fromVertices,
                                                     int sampleCount) {
        return getPosteriorSamples(bayesNet, fromVertices, sampleCount, KeanuRandom.getDefaultRandom());
    }

    public static NetworkSamples getPosteriorSamples(BayesianNetwork bayesNet,
                                                     List<Vertex<?>> fromVertices,
                                                     int sampleCount,
                                                     KeanuRandom random) {

        bayesNet.cascadeObservations();

        Map<Long, List<?>> samples = new HashMap<>();
        long acceptedCount = 0;

        while (acceptedCount < sampleCount) {
            sampleLatents(bayesNet.getLatentVertices(), random);
            if (matchesObservation(bayesNet.getObservedVertices())) {
                takeSamples(samples, fromVertices);
                acceptedCount++;
            }
        }

        return new NetworkSamples(samples, sampleCount);
    }

    private static void sampleLatents(List<? extends Vertex> latents, KeanuRandom random) {
        latents.forEach(vertex -> setFromSample((Vertex<?>) vertex, random));
    }

    private static <T> void setFromSample(Vertex<T> v, KeanuRandom random) {
        v.setAndCascade(v.sample(random));
    }

    private static boolean matchesObservation(List<? extends Vertex> observedVertices) {
        return observedVertices.stream()
            .allMatch(v -> v.matchesObservation());
    }

    private static void takeSamples(Map<Long, List<?>> samples, List<? extends Vertex<?>> fromVertices) {
        fromVertices.forEach(vertex -> addSampleForVertex(vertex, samples));
    }

    private static <T> void addSampleForVertex(Vertex<T> vertex, Map<Long, List<?>> samples) {
        List<T> samplesForVertex = (List<T>) samples.computeIfAbsent(vertex.getId(), v -> new ArrayList<T>());
        samplesForVertex.add(vertex.getValue());
    }
}
