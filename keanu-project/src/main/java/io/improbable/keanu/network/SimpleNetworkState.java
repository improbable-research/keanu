package io.improbable.keanu.network;

import io.improbable.keanu.vertices.Vertex;

import java.util.Map;
import java.util.Set;

public class SimpleNetworkState implements NetworkState {

    private final Map<Long, ?> vertexValues;

    public SimpleNetworkState(Map<Long, ?> vertexValues) {
        this.vertexValues = vertexValues;
    }

    @Override
    public <T> T get(Vertex<T> vertex) {
        return (T) vertexValues.get(vertex.getId());
    }

    @Override
    public <T> T get(long vertexId) {
        return (T) vertexValues.get(vertexId);
    }

    @Override
    public Set<Long> getVertexIds() {
        return vertexValues.keySet();
    }
}
