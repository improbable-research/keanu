package io.improbable.keanu.algorithms.sampling;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.graphtraversal.TopologicalSort;
import io.improbable.keanu.network.BayesNet;
import io.improbable.keanu.vertices.Vertex;

import java.util.*;

public class Prior {

    private Prior() {
    }

    public static NetworkSamples sample(BayesNet bayesNet,
                                        List<? extends Vertex> fromVertices,
                                        int sampleCount,
                                        Random random) {

        if (!bayesNet.getObservedVertices().isEmpty()) {
            throw new IllegalStateException("Cannot sample prior from graph with observations");
        }

        List<? extends Vertex> topologicallySorted = TopologicalSort.sort(bayesNet.getLatentVertices());
        Map<String, List> samplesByVertex = new HashMap<>();

        for (int sampleNum = 0; sampleNum < sampleCount; sampleNum++) {
            nextSample(topologicallySorted, random);
            takeSamples(samplesByVertex, fromVertices);
        }

        return new NetworkSamples(samplesByVertex, sampleCount);
    }

    private static void nextSample(List<? extends Vertex> topologicallySorted, Random random) {
        for (Vertex<?> vertex : topologicallySorted) {
            setAndCascadeFromSample(vertex, random);
        }
    }

    private static <T> void setAndCascadeFromSample(Vertex<T> vertex, Random random) {
        vertex.setAndCascade(vertex.sample(random));
    }

    private static void takeSamples(Map<String, List> samples, List<? extends Vertex> fromVertices) {
        fromVertices.forEach(vertex -> addSampleForVertex(vertex, samples));
    }

    private static void addSampleForVertex(Vertex vertex, Map<String, List> samples) {
        List samplesForVertex = samples.computeIfAbsent(vertex.getId(), v -> new ArrayList<>());
        samplesForVertex.add(vertex.getValue());
    }
}
