package io.improbable.keanu.plating;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;

import io.improbable.keanu.vertices.ProxyVertex;
import io.improbable.keanu.vertices.Vertex;

public class Plate {
    private Map<String, Vertex<?>> contents;

    public Plate() {
        this.contents = new HashMap<>();
    }

    public <T extends Vertex<?>> T add(String name, T v) {
        contents.put(name, v);
        return v;
    }

    public <T> Vertex<T> get(String name) {
        return (Vertex<T>) contents.get(name);
    }

    public Collection<Vertex<?>> getProxyVertices() {
        return contents.values().stream().filter(v -> v instanceof ProxyVertex).collect(Collectors.toList());
    }
}
