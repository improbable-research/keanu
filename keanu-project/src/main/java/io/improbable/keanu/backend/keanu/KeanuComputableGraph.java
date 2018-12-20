package io.improbable.keanu.backend.keanu;

import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.algorithms.variational.optimizer.VariableReference;
import io.improbable.keanu.backend.ComputableGraph;
import io.improbable.keanu.vertices.Vertex;

import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import static java.util.Collections.singletonList;
import static java.util.stream.Collectors.toMap;

public class KeanuComputableGraph implements ComputableGraph {

    private final Map<VariableReference, Vertex> vertexLookup;

    public KeanuComputableGraph(Set<Vertex> vertices) {
        this.vertexLookup = vertices.stream()
            .collect(toMap(Vertex::getReference, v -> v));
    }

    @Override
    public <T> T compute(Map<VariableReference, ?> inputs, VariableReference output) {
        return (T) compute(inputs, singletonList(output)).get(output);
    }

    @Override
    public Map<VariableReference, ?> compute(Map<VariableReference, ?> inputs, Collection<VariableReference> outputs) {

        for (Map.Entry<VariableReference, ?> input : inputs.entrySet()) {
            vertexLookup.get(input.getKey()).setValue(input.getValue());
        }

        List<Vertex> outputVertices = outputs.stream()
            .map(vertexLookup::get)
            .collect(Collectors.toList());

        VertexValuePropagation.eval(outputVertices);

        return outputVertices.stream()
            .collect(toMap(Vertex::getReference, v -> (Object) v.getValue()));
    }

    @Override
    public <T> T getInput(VariableReference input) {
        return (T) vertexLookup.get(input).getValue();
    }

}
