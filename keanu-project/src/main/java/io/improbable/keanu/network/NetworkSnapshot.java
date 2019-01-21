package io.improbable.keanu.network;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexState;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Saves the state (value and observed) of a specified collection of variables.
 */
public class NetworkSnapshot {

    public static NetworkSnapshot create(Set<? extends Vertex> vertices) {
        return new NetworkSnapshot(vertices);
    }

    private final Map<Vertex, VertexState> vertexStates;

    private NetworkSnapshot(Collection<? extends Vertex> vertices) {
        vertexStates = new HashMap<>();
        for (Vertex v : vertices) {
            vertexStates.put(v, v.getState());
        }
    }

    /**
     * Revert the state of the network to the previously saved state
     */
    public void apply() {
        for (Vertex v : vertexStates.keySet()) {
            v.setState(vertexStates.get(v));
        }
    }

}
