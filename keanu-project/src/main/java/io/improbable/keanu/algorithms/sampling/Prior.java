package io.improbable.keanu.algorithms.sampling;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.graphtraversal.TopologicalSort;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Prior {

    private Prior() {
    }

    public static NetworkSamples sample(BayesianNetwork bayesNet,
                                        List<? extends Vertex> fromVertices,
                                        int sampleCount,
                                        KeanuRandom random) {

        if (!bayesNet.getObservedVertices().isEmpty()) {
            throw new IllegalStateException("Cannot sample prior from graph with observations");
        }

        bayesNet.cascadeObservations();

        List<? extends Vertex> topologicallySorted = TopologicalSort.sort(bayesNet.getLatentVertices());
        Map<Long, List> samplesByVertex = new HashMap<>();

        for (int sampleNum = 0; sampleNum < sampleCount; sampleNum++) {
            nextSample(topologicallySorted, random);
            takeSamples(samplesByVertex, fromVertices);
        }

        return new NetworkSamples(samplesByVertex, sampleCount);
    }

    private static void nextSample(List<? extends Vertex> topologicallySorted, KeanuRandom random) {
        for (Vertex<?> vertex : topologicallySorted) {
            setAndCascadeFromSample(vertex, random);
        }
    }

    private static <T> void setAndCascadeFromSample(Vertex<T> vertex, KeanuRandom random) {
        vertex.setAndCascade(vertex.sample(random));
    }

    private static void takeSamples(Map<Long, List> samples, List<? extends Vertex> fromVertices) {
        fromVertices.forEach(vertex -> addSampleForVertex(vertex, samples));
    }

    private static void addSampleForVertex(Vertex vertex, Map<Long, List> samples) {
        List samplesForVertex = samples.computeIfAbsent(vertex.getId(), v -> new ArrayList<>());
        samplesForVertex.add(vertex.getValue());
    }
}
