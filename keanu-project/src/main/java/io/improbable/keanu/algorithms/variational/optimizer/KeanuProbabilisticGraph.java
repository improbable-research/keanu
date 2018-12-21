package io.improbable.keanu.algorithms.variational.optimizer;

import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.backend.LogProbWithSample;
import io.improbable.keanu.backend.ProbabilisticGraph;
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
    public LogProbWithSample logProbWithSample(Map<VariableReference, ?> inputs, List<VariableReference> outputs) {

        double logProb = logProb(inputs);
        Map<VariableReference, Object> sample = outputs.stream()
            .collect(toMap(
                output -> output,
                output -> vertexLookup.get(output).getValue()
            ));

        return new LogProbWithSample(logProb, sample);
    }

    @Override
    public List<Vertex> getLatentVariables() {
        return bayesianNetwork.getLatentVertices();
    }

    @Override
    public Map<VariableReference, ?> getLatentVariablesValues() {
        return bayesianNetwork.getLatentVertices().stream()
            .collect(toMap(
                Vertex::getReference,
                Vertex::getValue)
            );
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
