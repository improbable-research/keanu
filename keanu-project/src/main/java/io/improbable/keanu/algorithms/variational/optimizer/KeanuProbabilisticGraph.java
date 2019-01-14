package io.improbable.keanu.algorithms.variational.optimizer;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.NetworkSnapshot;
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

    private final LambdaSectionSnapshot lambdaSectionSnapshot;

    public KeanuProbabilisticGraph(Set<Vertex> variables) {
        this(new BayesianNetwork(variables));
    }

    public KeanuProbabilisticGraph(BayesianNetwork bayesianNetwork) {

        this.vertexLookup = bayesianNetwork.getLatentOrObservedVertices().stream()
            .collect(toMap(Vertex::getId, v -> v));

        this.latentVertices = ImmutableList.copyOf(bayesianNetwork.getLatentVertices());
        this.observedVertices = ImmutableList.copyOf(bayesianNetwork.getObservedVertices());
        this.latentOrObservedVertices = ImmutableList.copyOf(bayesianNetwork.getLatentOrObservedVertices());
        lambdaSectionSnapshot = new LambdaSectionSnapshot(latentVertices);

    }

    @Override
    public double logProb(Map<VariableReference, ?> inputs) {
        cascadeValues(inputs);
        return ProbabilityCalculator.calculateLogProbFor(this.latentOrObservedVertices);
    }

    @Override
    public double downstreamLogProb(Set<? extends Variable> vertices) {
        return lambdaSectionSnapshot.logProb(vertices);
    }

    @Override
    public double logLikelihood(Map<VariableReference, ?> inputs) {
        cascadeValues(inputs);
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

    @Override
    public void cascadeUpdate(Set<? extends Variable> inputs) {
        VertexValuePropagation.cascadeUpdate((Set<Vertex>) inputs);
    }

    @Override
    public void cascadeFixedVariables() {
        VertexValuePropagation.cascadeUpdate(this.observedVertices);
    }

    @Override
    public NetworkSnapshot getSnapshotOfAllAffectedVariables(Set<? extends Variable> variables) {
        return NetworkSnapshot.create(lambdaSectionSnapshot.getAllVerticesAffectedBy(variables));
    }

    @Override
    public boolean isDeterministic() {
        return latentOrObservedVertices.isEmpty();
    }

    public void cascadeValues(Map<VariableReference, ?> inputs) {

        List<Vertex> updatedVertices = new ArrayList<>();
        for (Map.Entry<VariableReference, ?> input : inputs.entrySet()) {
            Vertex updatingVertex = vertexLookup.get(input.getKey());

            if (updatingVertex == null) {
                throw new IllegalArgumentException("Cannot cascade update for input: " + input.getKey());
            }

            updatingVertex.setValue(input.getValue());
            updatedVertices.add(updatingVertex);
        }

        cascadeUpdate(new HashSet<>(updatedVertices));
    }


}
