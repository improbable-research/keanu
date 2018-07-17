package io.improbable.keanu.network;

import io.improbable.keanu.vertices.Vertex;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Saves the value of a specified collection of vertices for later use.
 */
public class NetworkSnapshot {

    public static NetworkSnapshot create(Set<Vertex> vertices) {
        return new NetworkSnapshot(vertices);
    }

    private final Map<Vertex, Object> snapshot;

    private NetworkSnapshot(Collection<Vertex> vertices) {
        snapshot = new HashMap<>();
        for (Vertex v : vertices) {
            snapshot.put(v, v.getValue());
        }
    }

    /**
     * Revert the state of the network to the previously saved snapshot
     */
    public void apply() {
        for (Vertex v : snapshot.keySet()) {
            v.setValue(snapshot.get(v));
        }
    }

}
