package io.improbable.keanu.vertices;

import java.util.Map;

public interface VertexDictionary {

    <V extends IVertex<?>> V get(VertexLabel label);

    VertexDictionary withExtraEntries(Map<VertexLabel, IVertex<?>> extraEntries);

    static VertexDictionary backedBy(Map<VertexLabel, IVertex<?>> map) {
        return SimpleVertexDictionary.backedBy(map);
    }

    static VertexDictionary of(IVertex... vertices) {
        return SimpleVertexDictionary.of(vertices);
    }
}
