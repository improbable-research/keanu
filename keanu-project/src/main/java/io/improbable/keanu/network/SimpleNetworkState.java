package io.improbable.keanu.network;

import io.improbable.keanu.vertices.Vertex;

import java.util.Map;
import java.util.Set;

public class SimpleNetworkState implements NetworkState {

    private final Map<String, ?> vertexValues;

    public SimpleNetworkState(Map<String, ?> vertexValues) {
        this.vertexValues = vertexValues;
    }

    @Override
    public <T> T get(Vertex<T> vertex) {
        return (T) vertexValues.get(vertex.getId());
    }

    @Override
    public <T> T get(String vertexId) {
        return (T) vertexValues.get(vertexId);
    }

    @Override
    public Set<String> getVertexIds() {
        return vertexValues.keySet();
    }
}
