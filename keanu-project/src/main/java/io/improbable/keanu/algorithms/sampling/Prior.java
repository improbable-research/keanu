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

    /**
     * Samples from a Bayesian Network that only contains prior information. No observations can have been made.
     *<p>
     * Samples are taken by calculating a linear ordering of the network and cascading the sampled values
     * through the network in priority order.
     *
     * @param bayesNet the prior bayesian network to sample from
     * @param fromVertices the vertices to sample from
     * @param sampleCount the number of samples to take
     * @return prior samples of a bayesian network
     */
    public static NetworkSamples sample(BayesianNetwork bayesNet,
                                        List<? extends Vertex> fromVertices,
                                        int sampleCount) {
        return sample(bayesNet, fromVertices, sampleCount, KeanuRandom.getDefaultRandom());
    }

    public static NetworkSamples sample(BayesianNetwork bayesNet,
                                        List<? extends Vertex> fromVertices,
                                        int sampleCount,
                                        KeanuRandom random) {

        if (!bayesNet.getObservedVertices().isEmpty()) {
            throw new IllegalStateException("Cannot sample prior from graph with observations");
        }

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
