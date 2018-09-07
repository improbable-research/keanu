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
        VertexLabel label = scoped(v.getLabel());
        if (contents.containsKey(label)) {
            throw new IllegalArgumentException("Key " + label + " already exists");
        }
        contents.put(label, v);
        v.setLabel(label);
        return v;
    }

    private String getUniqueName() {
        return "Plate_" + this.hashCode();
    }

    private VertexLabel scoped(VertexLabel label) {
        return label.inNamespace(getUniqueName());
    }

    @Override
    public <T> Vertex<T> get(VertexLabel label) {
        Vertex<?> vertex = contents.get(label);
        if (vertex == null) {
            vertex = contents.get(scoped(label));
        }
        if (vertex == null) {
            throw new IllegalArgumentException("Cannot find VertexLabel " + label);
        }
        return (Vertex<T>) vertex;
    }

    public Collection<Vertex<?>> getProxyVertices() {
        return contents.values().stream().filter(v -> v instanceof ProxyVertex).collect(Collectors.toList());
    }
}
