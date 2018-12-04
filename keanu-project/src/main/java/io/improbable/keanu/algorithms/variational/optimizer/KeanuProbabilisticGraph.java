package io.improbable.keanu.algorithms.variational.optimizer;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.ProbabilityCalculator;
import io.improbable.keanu.vertices.Vertex;
import lombok.Getter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

import static java.util.stream.Collectors.toList;
import static java.util.stream.Collectors.toMap;

public class KeanuProbabilisticGraph implements ProbabilisticGraph {

    private final Map<VariableReference, Vertex> vertexLookup;

    @Getter
    private final List<Vertex> latentVertices;

    @Getter
    private final List<Vertex> observedVertices;

    @Getter
    private final List<Vertex> latentOrObservedVertices;

    public KeanuProbabilisticGraph(BayesianNetwork bayesianNetwork) {

        this.vertexLookup = bayesianNetwork.getLatentVertices().stream()
            .collect(toMap(Vertex::getId, v -> v));

        this.latentVertices = ImmutableList.copyOf(bayesianNetwork.getLatentVertices());
        this.observedVertices = ImmutableList.copyOf(bayesianNetwork.getObservedVertices());
        this.latentOrObservedVertices = ImmutableList.copyOf(bayesianNetwork.getLatentOrObservedVertices());
    }

    @Override
    public double logProb(Map<VariableReference, ?> inputs) {
        cascadeUpdate(inputs);
        return ProbabilityCalculator.calculateLogProbFor(this.latentOrObservedVertices);
    }

    @Override
    public double logLikelihood(Map<VariableReference, ?> inputs) {
        cascadeUpdate(inputs);
        return ProbabilityCalculator.calculateLogProbFor(this.observedVertices);
    }

    @Override
    public List<VariableReference> getLatentVariables() {
        return this.latentVertices.stream()
            .map(Vertex::getId)
            .collect(toList());
    }

    @Override
    public Map<VariableReference, ?> getLatentVariablesValues() {
        return latentVertices.stream()
            .collect(toMap(
                Vertex::getId,
                Vertex::getValue)
            );
    }

    @Override
    public Map<VariableReference, long[]> getLatentVariablesShapes() {
        Map<VariableReference, ?> latentVariablesValues = getLatentVariablesValues();
        return getLatentVariables().stream()
            .collect(Collectors.toMap(
                v -> v,
                v -> ((Tensor) latentVariablesValues.get(v)).getShape())
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

    /**
     * @param vertex to get unique string for
     * @return A string that is unique to this vertex within any graph that this
     * vertex is a member of.
     */
    public static String getUniqueStringReference(Vertex vertex) {
        if (vertex.getLabel() != null) {
            return vertex.getLabel().toString();
        } else {
            return Arrays.stream(vertex.getId().getValue()).boxed()
                .map(Objects::toString)
                .collect(Collectors.joining("_"));
        }
    }

}