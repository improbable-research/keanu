package io.improbable.keanu.algorithms;

import io.improbable.keanu.network.NetworkState;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import lombok.Value;

import java.util.Set;

/**
 * A network sample contains the state of the network (vertex values) and the logOfMasterP at
 * a given point in time.
 */
@Value
public class NetworkSample {

    private final NetworkState networkState;
    private final double logOfMasterP;

    /**
     * @param vertex the vertex to get the values of
     * @param <T> the type of the values that the vertex contains
     * @return the values of the specified vertex
     */
    public <T> T get(Vertex<T> vertex) {
        return networkState.get(vertex);
    }

    /**
     * @param vertexId the ID of the vertex to get the values of
     * @param <T> the type of the values that the vertex contains
     * @return the values of the specified vertex
     */
    public <T> T get(VertexId vertexId) {
        return networkState.get(vertexId);
    }

    public Set<VertexId> getVertexIds() {
        return networkState.getVertexIds();
    }
}
