package io.improbable.keanu.network;

import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.ProxyVertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.VertexLabel;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class ModelComposition {

    private ModelComposition() {
    }

    /**
     * Connects one model within an other, marking it such that it's clear this is a nested model
     * <p>
     * This function will take a previously constructed BayesNet and do a number of things:
     * - Hook up any Proxy Vertices to the parents specified in the inputVertices
     * - Increment the depth for all "internal" nodes and the BayesNet itself
     * - Pass back any specified output nodes from the Bayesnet (keeping them at the outer depth)
     * <p>
     * Output nodes will be returned unlabelled
     * </p>
     *
     * @param bayesianNetwork The Bayesian Network to compose in to the current model
     * @param inputVertices   The mapping from Proxy label to actual input vertex
     * @param desiredOutputs  The set of labels we wish to output from the supplied BayesNet
     * @return A map of Labels to Output Vertices
     */
    public static Map<VertexLabel, IVertex> composeModel(BayesianNetwork bayesianNetwork,
                                                         Map<VertexLabel, IVertex> inputVertices,
                                                         List<VertexLabel> desiredOutputs) {
        Map<VertexLabel, IVertex> outputMap = extractOutputs(bayesianNetwork, desiredOutputs);
        increaseDepth(bayesianNetwork, outputMap);
        checkAndLinkInputs(bayesianNetwork, inputVertices);
        cleanOutputLabels(outputMap);

        return outputMap;
    }

    private static Map<VertexLabel, IVertex> extractOutputs(BayesianNetwork bayesianNetwork,
                                                            List<VertexLabel> desiredOutputs) {
        if (desiredOutputs.isEmpty()) {
            throw new IllegalArgumentException("At least one output must be specified");
        }

        Map<VertexLabel, IVertex> outputMap = new HashMap<>();

        for (VertexLabel label : desiredOutputs) {
            IVertex v = bayesianNetwork.getVertexByLabel(label);
            if (v == null) {
                throw new IllegalArgumentException("Unable to find Output Vertex: " + label);
            }
            outputMap.put(label, v);
        }

        return outputMap;
    }

    private static void increaseDepth(BayesianNetwork bayesianNetwork, Map<VertexLabel, IVertex> outputVertices) {
        VertexId newPrefix = new VertexId();
        bayesianNetwork.incrementIndentation();
        bayesianNetwork.getVertices().stream()
            .filter(v -> !outputVertices.containsKey(v.getLabel()))
            .forEach(v -> v.getId().addPrefix(newPrefix));
        bayesianNetwork.getVertices().stream()
            .filter(v -> outputVertices.containsKey(v.getLabel()))
            .forEach(v -> v.getId().resetID());
    }

    private static void checkAndLinkInputs(BayesianNetwork bayesianNetwork, Map<VertexLabel, IVertex> inputs) {
        for (Map.Entry<VertexLabel, IVertex> entry : inputs.entrySet()) {
            IVertex v = bayesianNetwork.getVertexByLabel(entry.getKey());

            if (v == null) {
                throw new IllegalArgumentException("No node labelled \"" + entry.getKey() + "\" found");
            }

            if (v instanceof ProxyVertex) {
                ProxyVertex proxyVertex = (ProxyVertex) v;
                if (proxyVertex.hasParent()) {
                    throw new IllegalArgumentException("Proxy Vertex for \"" + v.getLabel() + "\" already has Parent");
                } else {
                    proxyVertex.setParent(entry.getValue());
                }
            } else {
                throw new IllegalArgumentException("Input node \"" + entry.getKey() + "\" is not a Proxy Vertex");
            }
        }
    }

    private static void cleanOutputLabels(Map<VertexLabel, IVertex> outputMap) {
        outputMap.values().stream()
            .forEach(v -> v.setLabel((VertexLabel) null));
    }

}
