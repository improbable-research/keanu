package io.improbable.keanu.algorithms.variational.optimizer.nongradient;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.algorithms.variational.optimizer.nongradient.ProbabilisticGraph;
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

    private final Map<String, Vertex> vertexLookup;

    @Getter
    private final List<Vertex> latentVertices;

    @Getter
    private final List<Vertex> observedVertices;

    @Getter
    private final List<Vertex> latentOrObservedVertices;

    public KeanuProbabilisticGraph(BayesianNetwork bayesianNetwork) {

        this.vertexLookup = bayesianNetwork.getLatentVertices().stream()
            .collect(toMap(Vertex::getUniqueStringReference, v -> v));

        this.latentVertices = ImmutableList.copyOf(bayesianNetwork.getLatentVertices());
        this.observedVertices = ImmutableList.copyOf(bayesianNetwork.getObservedVertices());
        this.latentOrObservedVertices = ImmutableList.copyOf(bayesianNetwork.getLatentOrObservedVertices());
    }

    @Override
    public double logProb(Map<String, ?> inputs) {
        cascadeUpdate(inputs);
        return ProbabilityCalculator.calculateLogProbFor(this.latentOrObservedVertices);
    }

    @Override
    public double logLikelihood(Map<String, ?> inputs) {
        cascadeUpdate(inputs);
        return ProbabilityCalculator.calculateLogProbFor(this.observedVertices);
    }

    @Override
    public List<String> getLatentVariables() {
        return ImmutableList.copyOf(
            this.latentVertices.stream()
                .map(Vertex::getUniqueStringReference)
                .collect(Collectors.toList())
        );
    }

    @Override
    public Map<String, ?> getLatentVariablesValues() {
        return latentVertices.stream()
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

            if (updatingVertex == null) {
                throw new IllegalArgumentException("Cannot cascade update for input: " + input.getKey());
            }

            updatingVertex.setValue(input.getValue());
            updatedVertices.add(updatingVertex);
        }

        VertexValuePropagation.cascadeUpdate(updatedVertices);
    }

}
