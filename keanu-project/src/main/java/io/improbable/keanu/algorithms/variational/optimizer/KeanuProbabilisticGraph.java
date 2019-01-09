package io.improbable.keanu.algorithms.variational.optimizer;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ProbabilityCalculator;
import io.improbable.keanu.vertices.Vertex;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import static java.util.stream.Collectors.toMap;

public class KeanuProbabilisticGraph implements ProbabilisticGraph {

    private final Map<VariableReference, Vertex> vertexLookup;

    private final List<Vertex> latentVertices;

    private final List<Vertex> observedVertices;

    private final List<Vertex> latentOrObservedVertices;

    public KeanuProbabilisticGraph(BayesianNetwork bayesianNetwork) {

        this.vertexLookup = bayesianNetwork.getLatentVertices().stream()
            .collect(toMap(Vertex::getId, v -> v));

        this.latentVertices = ImmutableList.copyOf(bayesianNetwork.getLatentVertices());
        this.observedVertices = ImmutableList.copyOf(bayesianNetwork.getObservedVertices());
        this.latentOrObservedVertices = ImmutableList.copyOf(bayesianNetwork.getLatentOrObservedVertices());
    }

    @Override
    public double logProb(List<? extends Variable> variables) {
        cascadeUpdate(variables);
        return ProbabilityCalculator.calculateLogProbFor(this.latentOrObservedVertices);
    }

    @Override
    public double logProbOfProbabilisticVertices() {
        return logProb(latentOrObservedVertices);
    }

    @Override
    public double logLikelihood(List<? extends Variable> variables) {
        cascadeUpdate(variables);
        return ProbabilityCalculator.calculateLogProbFor(this.observedVertices);
    }

    @Override
    public List<? extends Variable> getLatentVariables() {
        return this.latentVertices;
    }

    @Override
    public List<? extends Variable<DoubleTensor>> getContinuousLatentVariables() {
        return getLatentVariables().stream()
            .filter(v -> v.getValue() instanceof DoubleTensor)
            .map(v -> (Variable<DoubleTensor>) v)
            .collect(Collectors.toList());
    }

    public void cascadeUpdate(List<? extends Variable> inputs) {

        List<Vertex> updatedVertices = new ArrayList<>();
        for (Variable input : inputs) {
            Vertex updatingVertex = vertexLookup.get(input.getReference());

            if (updatingVertex == null) {
                throw new IllegalArgumentException("Cannot cascade update for input: " + input.getReference());
            }

            updatingVertex.setValue(input);
            updatedVertices.add(updatingVertex);
        }

        VertexValuePropagation.cascadeUpdate(updatedVertices);
    }
}
