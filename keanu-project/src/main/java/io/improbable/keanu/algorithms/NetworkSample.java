package io.improbable.keanu.algorithms;

import io.improbable.keanu.network.NetworkState;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import lombok.Getter;

import java.util.Set;

public class NetworkSample {

    @Getter
    private final NetworkState networkState;
    @Getter
    private final double logOfMasterP;

    public <T> T get(Vertex<T> vertex) {
        return networkState.get(vertex);
    }

    public <T> T get(VertexId vertexId) {
        return networkState.get(vertexId);
    }

    public Set<VertexId> getVertexIds() {
        return networkState.getVertexIds();
    }

    public NetworkSample(NetworkState networkState, double logOfMasterP) {
        this.networkState = networkState;
        this.logOfMasterP = logOfMasterP;
    }
}
