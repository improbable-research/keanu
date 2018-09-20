package io.improbable.keanu.vertices;

import java.util.Map;

public interface VertexDictionary {

    <V extends Vertex<?>> V get(VertexLabel label);

    VertexDictionary getAllVertices();

    static VertexDictionary backedBy(Map<VertexLabel, Vertex<?>> map) {
        return SimpleVertexDictionary.backedBy(map);
    }

    static VertexDictionary of(Vertex... vertices) {
        return SimpleVertexDictionary.of(vertices);
    }
}
