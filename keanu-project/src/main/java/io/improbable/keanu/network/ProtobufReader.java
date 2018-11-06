package io.improbable.keanu.network;

import io.improbable.keanu.KeanuSavedBayesNet;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;

import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;

public class ProtobufReader {
    public static BayesianNetwork loadNetwork(InputStream input) throws IOException {
        Map<KeanuSavedBayesNet.VertexID, Vertex> instantiatedVertices = new HashMap<>();
        KeanuSavedBayesNet.BayesianNetwork parsedNet = KeanuSavedBayesNet.BayesianNetwork.parseFrom(input);

        for (KeanuSavedBayesNet.Vertex vertex : parsedNet.getVerticesList()) {
            Vertex newVertex = Vertex.fromProtoBuf(vertex, instantiatedVertices);
            instantiatedVertices.put(vertex.getId(), newVertex);
        }

        BayesianNetwork bayesNet = new BayesianNetwork(instantiatedVertices.values());

        loadDefaultValues(parsedNet, instantiatedVertices, bayesNet);

        return bayesNet;
    }

    private static void loadDefaultValues(KeanuSavedBayesNet.BayesianNetwork parsedNet,
                                          Map<KeanuSavedBayesNet.VertexID, Vertex> instantiatedVertices,
                                          BayesianNetwork bayesNet) {
        for (KeanuSavedBayesNet.StoredValue value : parsedNet.getDefaultStateList()) {
            Vertex targetVertex = null;

            if (value.hasId()) {
                targetVertex = instantiatedVertices.get(value.getId());
            }

            if (value.getVertexLabel() != "") {
                Vertex newTarget = bayesNet.getVertexByLabel(new VertexLabel(value.getVertexLabel()));

                if (targetVertex != null && newTarget != targetVertex) {
                    throw new IllegalArgumentException("Label and VertexID don't refer to same Vertex");
                } else {
                    targetVertex = newTarget;
                }
            }

            if (targetVertex == null) {
                throw new IllegalArgumentException("Value specified for unknown Vertex");
            }

            targetVertex.setValue(value.getValue());
        }
    }
}
