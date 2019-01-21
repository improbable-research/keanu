package io.improbable.keanu.algorithms.variational.optimizer;

import io.improbable.keanu.network.LambdaSection;
import io.improbable.keanu.vertices.ProbabilityCalculator;
import io.improbable.keanu.vertices.Vertex;

import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public class LambdaSectionSnapshot {
    private static final boolean USE_CACHE_ON_REJECTION = true;

    private final List<Vertex> latentVertices;
    private Map<Variable, LambdaSection> affectedVariablesCache;

    public LambdaSectionSnapshot(List<Vertex> latentVertices) {
        this.latentVertices = latentVertices;
    }

    private void update() {
        if (affectedVariablesCache == null) {
            affectedVariablesCache = createVariablesAffectedByCache(
                latentVertices,
                USE_CACHE_ON_REJECTION
            );
        }
    }

    public double logProb(Set<? extends Variable> vertices) {
        update();
        double sumLogProb = 0.0;
        for (Variable v : vertices) {
            sumLogProb += ProbabilityCalculator.calculateLogProbFor(affectedVariablesCache.get(v).getLatentAndObservedVertices());
        }
        return sumLogProb;
    }

    public Set<? extends Variable> getAllVariablesAffectedBy(Set<? extends Variable> variables) {
        update();
        Set<Variable> allAffectedVertices = new HashSet<>();
        for (Variable variable : variables) {
            allAffectedVertices.addAll(affectedVariablesCache.get(variable).getAllVertices());
        }
        return allAffectedVertices;
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
    private static Map<Variable, LambdaSection> createVariablesAffectedByCache(List<Vertex> latentVertices,
                                                                               boolean useCacheOnRejection) {
        return latentVertices.stream()
            .collect(Collectors.toMap(
                v -> v,
                v -> LambdaSection.getDownstreamLambdaSection(v, useCacheOnRejection)
            ));
    }
}
