package io.improbable.keanu.algorithms.variational.optimizer;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.LambdaSection;
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

    private static final boolean USE_CACHE_ON_REJECTION = true;

    private final Map<VariableReference, Vertex> vertexLookup;

    private final List<Vertex> latentVertices;

    private final List<Vertex> observedVertices;

    private final List<Vertex> latentOrObservedVertices;

    private Map<Variable, LambdaSection> affectedVerticesCache;

    public KeanuProbabilisticGraph(BayesianNetwork bayesianNetwork) {

        this.vertexLookup = bayesianNetwork.getLatentOrObservedVertices().stream()
            .collect(toMap(Vertex::getId, v -> v));

        this.latentVertices = ImmutableList.copyOf(bayesianNetwork.getLatentVertices());
        this.observedVertices = ImmutableList.copyOf(bayesianNetwork.getObservedVertices());
        this.latentOrObservedVertices = ImmutableList.copyOf(bayesianNetwork.getLatentOrObservedVertices());
        this.affectedVerticesCache = null;
    }

    @Override
    public double logProb(Map<VariableReference, ?> inputs) {
        cascadeValues(inputs);
        return ProbabilityCalculator.calculateLogProbFor(this.latentOrObservedVertices);
    }

    @Override
    public double downstreamLogProb(Set<? extends Variable> vertices) {
        if (affectedVerticesCache == null) {
            affectedVerticesCache = createVerticesAffectedByCache(
                latentVertices,
                USE_CACHE_ON_REJECTION
            );
        }
        double sumLogProb = 0.0;
        for (Variable v : vertices) {
            sumLogProb += ProbabilityCalculator.calculateLogProbFor(affectedVerticesCache.get(v).getLatentAndObservedVertices());
        }
        return sumLogProb;
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
        VertexValuePropagation.cascadeUpdate((Vertex) inputs);
    }

    @Override
    public void cascadeFixedVariables() {
        VertexValuePropagation.cascadeUpdate(this.observedVertices);
    }

    @Override
    public NetworkSnapshot getSnapshotOfAllAffectedVariables(Set<? extends Variable> variables) {
        if (affectedVerticesCache == null) {
            affectedVerticesCache = createVerticesAffectedByCache(
                latentVertices,
                USE_CACHE_ON_REJECTION
            );
        }
        Set<Variable> allAffectedVertices = new HashSet<>();
        for (Variable variable : variables) {
            allAffectedVertices.addAll(affectedVerticesCache.get(variable).getAllVertices());
        }

        return NetworkSnapshot.create(allAffectedVertices);
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

    /**
     * This creates a cache of potentially all vertices downstream to an observed or probabilistic vertex
     * from each latent vertex. If useCacheOnRejection is false then only the downstream observed or probabilistic
     * is cached.
     *
     * @param latentVertices      The latent vertices to create a cache for
     * @param useCacheOnRejection Whether or not to cache the entire downstream set or just the observed/probabilistic
     * @return A vertex to Lambda Section map that represents the downstream Lambda Section for each latent vertex.
     * This Lambda Section may include all of the nonprobabilistic vertices if useCacheOnRejection is enabled.
     */
    private static Map<Variable, LambdaSection> createVerticesAffectedByCache(List<Vertex> latentVertices,
                                                                              boolean useCacheOnRejection) {
        return latentVertices.stream()
            .collect(Collectors.toMap(
                v -> v,
                v -> LambdaSection.getDownstreamLambdaSection(v, useCacheOnRejection)
            ));
    }
}
