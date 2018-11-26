package io.improbable.keanu.backend.keanu;

import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.backend.ProbabilisticGraph;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.ProbabilityCalculator;
import io.improbable.keanu.vertices.Vertex;
import lombok.Getter;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
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
        return ProbabilityCalculator.calculateLogProbFor(this.bayesianNetwork.getLatentOrObservedVertices());
    }

    @Override
    public double logLikelihood(Map<String, ?> inputs) {
        cascadeUpdate(inputs);
        return ProbabilityCalculator.calculateLogProbFor(this.bayesianNetwork.getObservedVertices());
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

    @Override
    public Map<String, long[]> getLatentVariablesShapes() {
        Map<String, ?> latentVariablesValues = getLatentVariablesValues();
        return getLatentVariables().stream()
            .collect(Collectors.toMap(
                v -> v,
                v -> ((Tensor) latentVariablesValues.get(v)).getShape())
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
