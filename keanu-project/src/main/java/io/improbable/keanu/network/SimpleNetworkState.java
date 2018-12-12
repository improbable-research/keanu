package io.improbable.keanu.network;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;

import java.util.Map;
import java.util.Set;

public class SimpleNetworkState implements NetworkState {

    private final Map<VertexId, ?> vertexValues;

    public SimpleNetworkState(Map<VertexId, ?> vertexValues) {
        this.vertexValues = vertexValues;
    }

    @Override
    public <T> T get(Vertex<T> vertex) {
        return (T) vertexValues.get(vertex.getId());
    }

    @Override
    public <T> T get(VertexId vertexId) {
        return (T) vertexValues.get(vertexId);
    }

    @Override
    public Set<VertexId> getVertexIds() {
        return vertexValues.keySet();
    }
}
