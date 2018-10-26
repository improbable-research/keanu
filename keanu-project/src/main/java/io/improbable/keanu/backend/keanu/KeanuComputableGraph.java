package io.improbable.keanu.backend.keanu;

import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.backend.ComputableGraph;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;

import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import static java.util.Collections.singletonList;
import static java.util.stream.Collectors.toMap;

public class KeanuComputableGraph implements ComputableGraph {

    private final Map<VertexLabel, Vertex> vertexLookup;

    public KeanuComputableGraph(Set<Vertex> vertices) {
        this.vertexLookup = vertices.stream()
            .filter(v -> v.getLabel() != null)
            .collect(toMap(Vertex::getLabel, v -> v));
    }

    @Override
    public <T> T compute(Map<String, ?> inputs, String output) {
        return (T) compute(inputs, singletonList(output)).get(output);
    }

    @Override
    public Map<String, ?> compute(Map<String, ?> inputs, Collection<String> outputs) {

        for (Map.Entry<String, ?> input : inputs.entrySet()) {
            vertexLookup.get(new VertexLabel(input.getKey())).setValue(input.getValue());
        }

        List<Vertex> outputVertices = outputs.stream()
            .map(label -> vertexLookup.get(new VertexLabel(label)))
            .collect(Collectors.toList());

        VertexValuePropagation.eval(outputVertices);

        return outputVertices.stream()
            .collect(toMap(v -> v.getLabel().toString(), v -> (Object) v.getValue()));
    }

}
