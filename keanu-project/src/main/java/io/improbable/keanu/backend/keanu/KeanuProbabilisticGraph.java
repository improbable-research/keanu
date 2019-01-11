package io.improbable.keanu.backend.keanu;

import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.backend.ProbabilisticGraph;
import io.improbable.keanu.backend.VariableReference;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.ProbabilityCalculator;
import io.improbable.keanu.vertices.Vertex;
import lombok.Getter;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static java.util.stream.Collectors.toMap;

public class KeanuProbabilisticGraph implements ProbabilisticGraph {

    @Getter
    private final BayesianNetwork bayesianNetwork;

    @Getter
    private final Map<VariableReference, Vertex> vertexLookup;

    public KeanuProbabilisticGraph(BayesianNetwork bayesianNetwork) {
        this.bayesianNetwork = bayesianNetwork;
        this.vertexLookup = bayesianNetwork.getVertices().stream()
            .collect(toMap(Vertex::getReference, v -> v));
    }

    @Override
    public double logProb(Map<VariableReference, ?> inputs) {
        cascadeUpdate(inputs);
        return ProbabilityCalculator.calculateLogProbFor(bayesianNetwork.getLatentOrObservedVertices());
    }

    @Override
    public double logLikelihood(Map<VariableReference, ?> inputs) {
        cascadeUpdate(inputs);
        return ProbabilityCalculator.calculateLogProbFor(bayesianNetwork.getObservedVertices());
    }

    @Override
    public List<Vertex> getLatentVariables() {
        return bayesianNetwork.getLatentVertices();
    }

    public void cascadeUpdate(Map<VariableReference, ?> inputs) {

        List<Vertex> updatedVertices = new ArrayList<>();
        for (Map.Entry<VariableReference, ?> input : inputs.entrySet()) {
            Vertex updatingVertex = vertexLookup.get(input.getKey());

            if (updatingVertex == null) {
                throw new IllegalArgumentException("Cannot cascade update for input: " + input.getKey());
            }

            updatingVertex.setValue(input.getValue());
            updatedVertices.add(updatingVertex);
        }

        VertexValuePropagation.cascadeUpdate(updatedVertices);
    }

}
