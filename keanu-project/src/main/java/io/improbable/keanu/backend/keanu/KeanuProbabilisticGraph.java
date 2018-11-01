package io.improbable.keanu.backend.keanu;

import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.backend.LogProbWithSample;
import io.improbable.keanu.backend.ProbabilisticGraph;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.Vertex;
import lombok.Getter;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import static java.util.stream.Collectors.toMap;

public class KeanuProbabilisticGraph implements ProbabilisticGraph {

    @Getter
    private final BayesianNetwork bayesianNetwork;

    @Getter
    private final Map<String, Vertex> vertexLookup;

    public KeanuProbabilisticGraph(BayesianNetwork bayesianNetwork) {
        this.bayesianNetwork = bayesianNetwork;
        this.vertexLookup = bayesianNetwork.getVertices().stream()
            .collect(toMap(Vertex::getUniqueStringReference, v -> v));
    }

    @Override
    public double logProb(Map<String, ?> inputs) {
        cascadeUpdate(inputs);
        return this.bayesianNetwork.getLogOfMasterP();
    }

    @Override
    public LogProbWithSample logProbWithSample(Map<String, ?> inputs, List<String> outputs) {

        double logProb = logProb(inputs);
        Map<String, Object> sample = outputs.stream()
            .collect(toMap(
                output -> output,
                output -> vertexLookup.get(output).getValue()
            ));

        return new LogProbWithSample(logProb, sample);
    }

    @Override
    public List<String> getLatentVariables() {
        return bayesianNetwork.getLatentVertices().stream()
            .map(Vertex::getUniqueStringReference)
            .collect(Collectors.toList());
    }

    @Override
    public Map<String, ?> getLatentVariablesValues() {
        return bayesianNetwork.getLatentVertices().stream()
            .collect(toMap(
                Vertex::getUniqueStringReference,
                Vertex::getValue)
            );
    }

    public void cascadeUpdate(Map<String, ?> inputs) {

        List<Vertex> updatedVertices = new ArrayList<>();
        for (Map.Entry<String, ?> input : inputs.entrySet()) {
            Vertex updatingVertex = vertexLookup.get(input.getKey());
            updatingVertex.setValue(input.getValue());
            updatedVertices.add(updatingVertex);
        }

        VertexValuePropagation.cascadeUpdate(updatedVertices);
    }

}
