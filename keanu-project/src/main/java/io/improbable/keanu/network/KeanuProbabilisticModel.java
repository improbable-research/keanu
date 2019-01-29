package io.improbable.keanu.network;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import io.improbable.keanu.algorithms.ProbabilisticModel;
import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ProbabilityCalculator;
import io.improbable.keanu.vertices.Vertex;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import static java.util.stream.Collectors.toMap;

/**
 * An implementation of {@link ProbabilisticModel} that is backed by a {@link BayesianNetwork}
 */
public class KeanuProbabilisticModel implements ProbabilisticModel {

    private final Map<VariableReference, Vertex> vertexLookup;

    private final List<Vertex> latentVertices;

    private final List<Vertex> observedVertices;

    private final List<Vertex> latentOrObservedVertices;
    private final LambdaSectionSnapshot lambdaSectionSnapshot;

    public KeanuProbabilisticModel(Collection<? extends Vertex> variables) {
        this(new BayesianNetwork(variables));
    }

    public KeanuProbabilisticModel(BayesianNetwork bayesianNetwork) {
        this.vertexLookup = bayesianNetwork.getLatentOrObservedVertices().stream()
            .collect(toMap(Vertex::getId, v -> v));

        this.latentVertices = ImmutableList.copyOf(bayesianNetwork.getLatentVertices());
        this.observedVertices = ImmutableList.copyOf(bayesianNetwork.getObservedVertices());
        this.latentOrObservedVertices = ImmutableList.copyOf(bayesianNetwork.getLatentOrObservedVertices());
        this.lambdaSectionSnapshot = new LambdaSectionSnapshot(latentVertices);

        resetModelToObservedState();
        checkBayesNetInHealthyState();
    }

    @Override
    public double logProb(Map<VariableReference, ?> inputs) {
        cascadeValues(inputs);
        return ProbabilityCalculator.calculateLogProbFor(this.latentOrObservedVertices);
    }

    @Override
    public double logProbAfter(Map<VariableReference, Object> newValues, double logProbBefore) {
        ImmutableSet.Builder<Vertex> affectedVerticesBuilder = ImmutableSet.builder();
        for (VariableReference reference : newValues.keySet()) {
            Vertex vertex = vertexLookup.get(reference);
            affectedVerticesBuilder.add(vertex);
        }
        Set<Vertex> affectedVertices = affectedVerticesBuilder.build();

        double lambdaSectionLogProbBefore = lambdaSectionSnapshot.logProb(affectedVertices);
        cascadeValues(newValues);
        double lambdaSectionLogProbAfter = lambdaSectionSnapshot.logProb(affectedVertices);
        double deltaLogProb = lambdaSectionLogProbAfter - lambdaSectionLogProbBefore;
        return logProbBefore + deltaLogProb;
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

    public List<Vertex> getLatentVertices() {
        return this.latentVertices;
    }

    public List<Vertex> getLatentOrObservedVertices() {
        return latentOrObservedVertices;
    }

    @Override
    public List<? extends Variable<DoubleTensor, ?>> getContinuousLatentVariables() {
        return getLatentVariables().stream()
            .filter(v -> v.getValue() instanceof DoubleTensor)
            .map(v -> (Variable<DoubleTensor, ?>) v)
            .collect(Collectors.toList());
    }

    private void checkBayesNetInHealthyState() {
        if (latentOrObservedVertices.isEmpty()) {
            throw new IllegalArgumentException("Cannot run inference or sampling from a completely deterministic BayesNet");
        } else if (ProbabilityCalculator.isImpossibleLogProb(this.logProb())) {
            throw new IllegalArgumentException("Cannot start optimizer on zero probability network");
        }
    }

    private void resetModelToObservedState() {
        VertexValuePropagation.cascadeUpdate(this.observedVertices);
    }

    protected void cascadeValues(Map<VariableReference, ?> inputs) {

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
