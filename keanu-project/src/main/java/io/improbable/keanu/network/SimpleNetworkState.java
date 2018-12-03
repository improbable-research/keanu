package io.improbable.keanu.network;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;

import java.util.Map;
import java.util.Optional;
import java.util.Set;

public class SimpleNetworkState implements NetworkState {

    private final Map<VertexId, ?> vertexValues;
    private final Optional<Double> logOfMasterP;

    public SimpleNetworkState(Map<VertexId, ?> vertexValues, double logOfMasterP) {
        this.vertexValues = vertexValues;
        this.logOfMasterP = Optional.of(logOfMasterP);
    }

    public SimpleNetworkState(Map<VertexId, ?> vertexValues) {
        this.vertexValues = vertexValues;
        this.logOfMasterP = Optional.empty();
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

    @Override
    public double getLogOfMasterP() {
        return logOfMasterP.orElseThrow(() -> new IllegalArgumentException("Network state doesn't have LogOfMasterP."));
    }
}
