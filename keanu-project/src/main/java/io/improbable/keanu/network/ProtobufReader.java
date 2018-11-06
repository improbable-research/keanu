package io.improbable.keanu.network;

import io.improbable.keanu.KeanuSavedBayesNet;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;

import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Constructor;
import java.util.HashMap;
import java.util.Map;

public class ProtobufReader {
    public static BayesianNetwork loadNetwork(InputStream input) throws IOException {
        Map<KeanuSavedBayesNet.VertexID, Vertex> instantiatedVertices = new HashMap<>();
        KeanuSavedBayesNet.BayesianNetwork parsedNet = KeanuSavedBayesNet.BayesianNetwork.parseFrom(input);

        for (KeanuSavedBayesNet.Vertex vertex : parsedNet.getVerticesList()) {
            Vertex newVertex = fromProtoBuf(vertex, instantiatedVertices);
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

    private static <T> Vertex<T> fromProtoBuf(KeanuSavedBayesNet.Vertex vertex,
                                             Map<KeanuSavedBayesNet.VertexID, Vertex> existingVertices) {
        Class vertexClass;
        try {
            vertexClass = Class.forName(vertex.getVertexType());
        } catch (ClassNotFoundException e) {
            throw new IllegalArgumentException("Unknown Vertex Type Specified: " + vertex.getVertexType(), e);
        }

        Constructor vertexConstructor;
        try {
            vertexConstructor = vertexClass.getConstructor(Map.class, KeanuSavedBayesNet.VertexValue.class);
        } catch (NoSuchMethodException e) {
            throw new
                IllegalArgumentException("Vertex Type doesn't support loading from Proto: " + vertex.getVertexType(), e);
        }

        Vertex<T> newVertex;
        Map<String, Vertex> parentsMap = getParentsMap(vertex, existingVertices);

        try {
            newVertex = (Vertex<T>)vertexConstructor.newInstance(parentsMap, vertex.getConstantValue());
        } catch (Exception e) {
            throw new IllegalArgumentException("Failed to create Vertex of Type: " + vertex.getVertexType(), e);
        }

        return newVertex;
    }

    private static Map<String, Vertex> getParentsMap(KeanuSavedBayesNet.Vertex vertex,
                                                     Map<KeanuSavedBayesNet.VertexID, Vertex> existingVertices) {
        Map<String, Vertex> parentsMap = new HashMap<>();

        for (KeanuSavedBayesNet.NamedParent namedParent : vertex.getParentsList()) {
            Vertex existingParent = existingVertices.get(namedParent.getId());
            if (existingParent == null) {
                throw new IllegalArgumentException("Invalid Parent Specified: " + namedParent);
            }

            parentsMap.put(namedParent.getName(), existingParent);
        }

        return parentsMap;
    }
}
