package io.improbable.keanu.vertices;

import io.improbable.keanu.network.NetworkSaver;

public interface NonSaveableVertex {
    default void save(NetworkSaver netSaver) {
        throw new IllegalArgumentException("Trying to save a vertex that isn't Saveable");
    }
}
