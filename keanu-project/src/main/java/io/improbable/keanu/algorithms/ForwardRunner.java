package io.improbable.keanu.algorithms;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.vertices.RandomVariable;

public class ForwardRunner {

    private ForwardRunner() {
    }

    /**
     * Samples from a Bayesian Network that only contains prior information. No observations can have been made.
     * Samples are taken by calculating a linear ordering of the network and cascading the sampled values
     * through the network in priority order.
     *
     * @param model the prior bayesian network to sample from
     * @param fromVertices the vertices to sample from
     * @param sampleCount the number of samples to take
     * @return prior samples of a bayesian network
     */
    public static NetworkSamples sample(ProbabilisticModel model,
                                        List<? extends RandomVariable> fromVertices,
                                        int sampleCount) {
        return sample(model, fromVertices, sampleCount, KeanuRandom.getDefaultRandom());
    }

    public static NetworkSamples sample(ProbabilisticModel model,
                                        List<? extends RandomVariable> fromVertices,
                                        int sampleCount,
                                        KeanuRandom random) {

        List<? extends RandomVariable> sorted = model.sort(model.getLatentVariables());
        Map<VariableReference, List> samplesByVertex = new HashMap<>();

        for (int sampleNum = 0; sampleNum < sampleCount; sampleNum++) {
            nextSample(sorted, random);
            takeSamples(samplesByVertex, fromVertices);
        }

        return new NetworkSamples(samplesByVertex, sampleCount);
    }

    private static void nextSample(List<? extends RandomVariable> topologicallySorted, KeanuRandom random) {
        for (RandomVariable variable: topologicallySorted) {
            setAndCascadeFromSample(variable, random);
        }
    }

    private static void setAndCascadeFromSample(RandomVariable variable, KeanuRandom random) {
        variable.setAndCascade(variable.sample(random));
    }

    private static void takeSamples(Map<VariableReference, List> samples, List<? extends RandomVariable> fromVertices) {
        fromVertices.forEach(vertex -> addSampleForVertex(vertex, samples));
    }

    private static void addSampleForVertex(RandomVariable vertex, Map<VariableReference, List> samples) {
        List samplesForVertex = samples.computeIfAbsent(vertex.getReference(), v -> new ArrayList<>());
        samplesForVertex.add(vertex.getValue());
    }
}