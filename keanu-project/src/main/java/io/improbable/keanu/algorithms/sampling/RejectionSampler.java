package io.improbable.keanu.algorithms.sampling;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.network.BayesNet;
import io.improbable.keanu.vertices.Vertex;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

public class RejectionSampler {

    private RejectionSampler() {
    }

    public static double getPosteriorProbability(List<? extends Vertex> latentVertices, List<Vertex> observedVertices, Supplier<Boolean> isSuccess, int sampleCount) {
        int matchedSampleCount = 0;
        int success = 0;

        while (matchedSampleCount < sampleCount) {
            sampleLatents(latentVertices);
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

    public static NetworkSamples getPosteriorSamples(BayesNet bayesNet, List<Vertex<?>> fromVertices, int sampleCount) {

        Map<String, List<?>> samples = new HashMap<>();
        long acceptedCount = 0;

        while (acceptedCount < sampleCount) {
            sampleLatents(bayesNet.getLatentVertices());
            if (matchesObservation(bayesNet.getObservedVertices())) {
                takeSamples(samples, fromVertices);
                acceptedCount++;
            }
        }

        return new NetworkSamples(samples, sampleCount);
    }

    private static void sampleLatents(List<? extends Vertex> latents) {
        latents.forEach(RejectionSampler::setFromSample);
    }

    private static <T> void setFromSample(Vertex<T> v) {
        v.setAndCascade(v.sample());
    }

    private static boolean matchesObservation(List<Vertex> observedVertices) {
        return observedVertices.stream()
                .allMatch(v -> v.densityAtValue() != 0.0);
    }

    private static void takeSamples(Map<String, List<?>> samples, List<? extends Vertex<?>> fromVertices) {
        fromVertices.forEach(vertex -> addSampleForVertex(vertex, samples));
    }

    private static <T> void addSampleForVertex(Vertex<T> vertex, Map<String, List<?>> samples) {
        List<T> samplesForVertex = (List<T>) samples.computeIfAbsent(vertex.getId(), v -> new ArrayList<T>());
        samplesForVertex.add(vertex.getValue());
    }
}
