package io.improbable.keanu.backend.keanu;

import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.backend.ComputableGraph;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.Vertex;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static java.util.stream.Collectors.toMap;

/**
 * A computable graph that wraps the Keanu vertices
 */
public class KeanuComputableGraph implements ComputableGraph {

    private final Map<VariableReference, Vertex> vertexLookup;
    private final List<Vertex> topoSortedGraph;
    private final Set<Vertex> outputs;

    public KeanuComputableGraph(List<Vertex> topoSortedGraph, Set<Vertex> outputs) {
        this.topoSortedGraph = new ArrayList<>(topoSortedGraph);
        this.outputs = outputs;

        this.vertexLookup = topoSortedGraph.stream()
            .collect(toMap(Vertex::getReference, v -> v));
    }

    @Override
    public Map<VariableReference, ?> compute(Map<VariableReference, ?> inputs) {

        for (Map.Entry<VariableReference, ?> input : inputs.entrySet()) {
            vertexLookup.get(input.getKey()).setValue(input.getValue());
        }

        for (int i = 0; i < topoSortedGraph.size(); i++) {

            final Vertex vertex = topoSortedGraph.get(i);

            if (!vertex.isProbabilistic() && !vertex.isObserved()) {
                vertex.setValue(((NonProbabilistic) vertex).calculate());
            }
        }

        return outputs.stream()
            .collect(toMap(Vertex::getReference, v -> (Object) v.getValue()));
    }

    @Override
    public <T> T getInput(VariableReference input) {
        return (T) vertexLookup.get(input).getValue();
    }

}
