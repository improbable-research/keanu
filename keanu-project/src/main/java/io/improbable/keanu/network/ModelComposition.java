package io.improbable.keanu.network;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import io.improbable.keanu.vertices.ProxyVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.VertexLabel;

public final class ModelComposition {

    private ModelComposition() { }

    public static Map<VertexLabel, Vertex> createModelVertices(BayesianNetwork bayesianNetwork,
                                                               Map<VertexLabel, Vertex> inputVertices,
                                                               List<VertexLabel> desiredOutputs) {
        Map<VertexLabel, Vertex> outputMap = extractOutputs(bayesianNetwork, desiredOutputs);
        increaseDepth(bayesianNetwork, outputMap);
        checkAndLinkInputs(bayesianNetwork, inputVertices);

        return outputMap;
    }

    private static Map<VertexLabel, Vertex> extractOutputs(BayesianNetwork bayesianNetwork,
                                                           List<VertexLabel> desiredOutputs) {
        if (desiredOutputs.size() == 0) {
            throw new IllegalArgumentException("At least one output must be specified");
        }

        Map<VertexLabel, Vertex> outputMap = new HashMap<>();

        for (VertexLabel label: desiredOutputs) {
            Vertex v = bayesianNetwork.getVertexByLabel(label);
            if (v == null) {
                throw new IllegalArgumentException("Unable to find Output Vertex: " + label);
            }
            outputMap.put(label, v);
        }

        return outputMap;
    }

    private static void increaseDepth(BayesianNetwork bayesianNetwork, Map<VertexLabel, Vertex> outputVertices) {
        VertexId newPrefix = new VertexId();
        bayesianNetwork.incrementDepth();
        bayesianNetwork.getAllVertices().stream()
            .filter(v -> !outputVertices.containsKey(v.getLabel()))
            .forEach(v -> v.getId().increaseDepth(newPrefix));
        bayesianNetwork.getAllVertices().stream()
            .filter(v -> outputVertices.containsKey(v.getLabel()))
            .forEach(v -> v.getId().resetID());
    }

    private static void checkAndLinkInputs(BayesianNetwork bayesianNetwork, Map<VertexLabel, Vertex> inputs) {
        for (Map.Entry<VertexLabel, Vertex> entry: inputs.entrySet()) {
            Vertex v = bayesianNetwork.getVertexByLabel(entry.getKey());

            if (v == null) {
                throw new IllegalArgumentException("No node labelled \"" + entry.getKey() + "\" found");
            }

            if (v instanceof ProxyVertex) {
                ProxyVertex proxyVertex = (ProxyVertex)v;
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

}
