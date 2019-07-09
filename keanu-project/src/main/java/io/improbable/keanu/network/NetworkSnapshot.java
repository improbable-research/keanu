package io.improbable.keanu.network;

import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.VertexState;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Saves the state (value and observed) of a specified collection of variables.
 */
public class NetworkSnapshot {

    public static NetworkSnapshot create(Set<? extends IVertex> vertices) {
        return new NetworkSnapshot(vertices);
    }

    private final Map<IVertex, VertexState> vertexStates;

    private NetworkSnapshot(Collection<? extends IVertex> vertices) {
        vertexStates = new HashMap<>();
        for (IVertex v : vertices) {
            vertexStates.put(v, v.getState());
        }
    }

    /**
     * Revert the state of the network to the previously saved state
     */
    public void apply() {
        for (IVertex v : vertexStates.keySet()) {
            v.setState(vertexStates.get(v));
        }
    }

}
