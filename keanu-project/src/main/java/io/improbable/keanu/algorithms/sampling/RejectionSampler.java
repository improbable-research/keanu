package io.improbable.keanu.algorithms.sampling;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.network.BayesNet;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

import java.util.*;
import java.util.function.Supplier;

public class RejectionSampler {

    private RejectionSampler() {
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

    public static NetworkSamples getPosteriorSamples(BayesNet bayesNet,
                                                     List<Vertex<?>> fromVertices,
                                                     int sampleCount) {
        return getPosteriorSamples(bayesNet, fromVertices, sampleCount, KeanuRandom.getDefaultRandom());
    }

    public static NetworkSamples getPosteriorSamples(BayesNet bayesNet,
                                                     List<Vertex<?>> fromVertices,
                                                     int sampleCount,
                                                     KeanuRandom random) {

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
            .allMatch(v -> v.logProbAtValue() != Double.NEGATIVE_INFINITY);
    }

    private static void takeSamples(Map<Long, List<?>> samples, List<? extends Vertex<?>> fromVertices) {
        fromVertices.forEach(vertex -> addSampleForVertex(vertex, samples));
    }

    private static <T> void addSampleForVertex(Vertex<T> vertex, Map<Long, List<?>> samples) {
        List<T> samplesForVertex = (List<T>) samples.computeIfAbsent(vertex.getId(), v -> new ArrayList<T>());
        samplesForVertex.add(vertex.getValue());
    }
}
