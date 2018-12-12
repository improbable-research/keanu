package io.improbable.keanu.algorithms;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import lombok.AllArgsConstructor;
import lombok.Getter;

import java.util.Map;
import java.util.Set;

/**
 * A network sample contains the state of the network (vertex values) and the logOfMasterP at
 * a given point in time.
 */
@AllArgsConstructor
public class NetworkSample {

    private final Map<VertexId, ?> vertexValues;

    @Getter
    private final double logOfMasterP;

    /**
     * @param vertex the vertex to get the values of
     * @param <T>    the type of the values that the vertex contains
     * @return the values of the specified vertex
     */
    public <T> T get(Vertex<T> vertex) {
        return (T) vertexValues.get(vertex.getId());
    }

    /**
     * @param vertexId the ID of the vertex to get the values of
     * @param <T>      the type of the values that the vertex contains
     * @return the values of the specified vertex
     */
    public <T> T get(VertexId vertexId) {
        return (T) vertexValues.get(vertexId);
    }

    public Set<VertexId> getVertexIds() {
        return vertexValues.keySet();
    }
}
