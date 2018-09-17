package io.improbable.keanu.plating;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import com.google.common.collect.ImmutableList;

import io.improbable.keanu.vertices.ProxyVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexDictionary;
import io.improbable.keanu.vertices.VertexLabel;

public class Plate implements VertexDictionary {

    private static final String NAME_PREFIX = "Plate_";
    private static Pattern NAME_REGEX = Pattern.compile(NAME_PREFIX + "-?[\\d]+$");

    private Map<VertexLabel, Vertex<?>> contents;

    public Plate() {
        this.contents = new HashMap<>();
    }


    public <T extends Vertex<?>> void addAll(T... vertices) {
        addAll(ImmutableList.copyOf(vertices));
    }

    public <T extends Vertex<?>> void addAll(Collection<T> vertices) {
        for (T v : vertices) {
            add(v);
        }
    }

    public <T extends Vertex<?>> T add(T v) {
        VertexLabel label = v.getLabel();
        if (label == null) {
            throw new PlateException("Vertex " + v + " must contain a label in order to be added to a plate");
        }
        String outerNamespace = label.getOuterNamespace().orElse("");
        if (NAME_REGEX.matcher(outerNamespace).matches()) {
            throw new PlateException("Vertex " + v + " has already been added to " + outerNamespace);
        }
        label = scoped(label);
        if (contents.containsKey(label)) {
            throw new IllegalArgumentException("Key " + label + " already exists");
        }
        contents.put(label, v);
        v.setLabel(label);
        return v;
    }

    private String getUniqueName() {
        return NAME_PREFIX + this.hashCode();
    }

    private VertexLabel scoped(VertexLabel label) {
        return label.withExtraNamespace(getUniqueName());
    }

    @Override
    public <V extends Vertex<?>> V get(VertexLabel label) {
        Vertex<?> vertex = contents.getOrDefault(label, contents.get(scoped(label)));

        if (vertex == null) {
            throw new IllegalArgumentException("Cannot find VertexLabel " + label);
        }
        return (V) vertex;
    }

    public Collection<Vertex<?>> getProxyVertices() {
        return contents.values().stream()
            .filter(v -> v instanceof ProxyVertex)
            .collect(Collectors.toList());
    }
}
