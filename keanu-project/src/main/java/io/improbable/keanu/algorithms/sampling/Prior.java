package io.improbable.keanu.algorithms.sampling;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.graphtraversal.TopologicalSort;
import io.improbable.keanu.network.BayesNet;
import io.improbable.keanu.vertices.Vertex;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Prior {

    private Prior() {
    }

    public static NetworkSamples sample(BayesNet bayesNet, List<? extends Vertex> fromVertices, int sampleCount) {

        if (!bayesNet.getObservedVertices().isEmpty()) {
            throw new IllegalStateException("Cannot sample prior from graph with observations");
        }

        List<? extends Vertex> topologicallySorted = TopologicalSort.sort(bayesNet.getLatentVertices());
        Map<Long, List> samplesByVertex = new HashMap<>();

        for (int sampleNum = 0; sampleNum < sampleCount; sampleNum++) {
            nextSample(topologicallySorted);
            takeSamples(samplesByVertex, fromVertices);
        }

        return new NetworkSamples(samplesByVertex, sampleCount);
    }

    private static void nextSample(List<? extends Vertex> topologicallySorted) {
        for (Vertex<?> vertex : topologicallySorted) {
            setAndCascadeFromSample(vertex);
        }
    }

    private static <T> void setAndCascadeFromSample(Vertex<T> vertex) {
        vertex.setAndCascade(vertex.sample());
    }

    private static void takeSamples(Map<Long, List> samples, List<? extends Vertex> fromVertices) {
        fromVertices.forEach(vertex -> addSampleForVertex(vertex, samples));
    }

    private static void addSampleForVertex(Vertex vertex, Map<Long, List> samples) {
        List samplesForVertex = samples.computeIfAbsent(vertex.getId(), v -> new ArrayList<>());
        samplesForVertex.add(vertex.getValue());
    }
}
