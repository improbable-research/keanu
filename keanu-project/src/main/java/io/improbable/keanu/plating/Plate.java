package io.improbable.keanu.plating;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;

import io.improbable.keanu.vertices.ProxyVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexDictionary;
import io.improbable.keanu.vertices.VertexLabel;

public class Plate implements VertexDictionary {
    private Map<VertexLabel, Vertex<?>> contents;

    public Plate() {
        this.contents = new HashMap<>();
    }

    public <T extends Vertex<?>> T add(T v) {
        contents.put(v.getLabel(), v);
        return v;
    }

    @Override
    public <T> Vertex<T> get(VertexLabel label) {
        return (Vertex<T>) contents.get(label);
    }

    public Collection<Vertex<?>> getProxyVertices() {
        return contents.values().stream().filter(v -> v instanceof ProxyVertex).collect(Collectors.toList());
    }
}
