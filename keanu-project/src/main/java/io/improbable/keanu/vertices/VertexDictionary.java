package io.improbable.keanu.vertices;

import java.util.Map;

public interface VertexDictionary {
    <T, V extends Vertex<T>> V get(VertexLabel label);

    static VertexDictionary backedBy(Map<VertexLabel, Vertex<?>> map) {
        return SimpleVertexDictionary.backedBy(map);
    }

    static VertexDictionary of(Vertex... vertices) {
        return SimpleVertexDictionary.of(vertices);
    }
}
