package io.improbable.keanu.vertices;

import io.improbable.keanu.network.NetworkWriter;

public interface NonSaveableVertex {
    default void save(NetworkWriter netWriter) {
        throw new IllegalArgumentException("Trying to save a vertex that isn't Saveable");
    }
    default void saveValue(NetworkWriter netWriter) {
        throw new IllegalArgumentException("Trying to save a vertex that isn't Saveable");
    }
}
