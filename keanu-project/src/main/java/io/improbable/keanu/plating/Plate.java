package io.improbable.keanu.plating;

import io.improbable.keanu.vertices.Vertex;

import java.util.HashMap;
import java.util.Map;

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
}
