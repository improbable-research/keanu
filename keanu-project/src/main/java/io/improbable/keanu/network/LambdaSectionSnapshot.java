package io.improbable.keanu.network;

import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.vertices.ProbabilityCalculator;
import io.improbable.keanu.vertices.Vertex;

import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * A snapshot of {@link LambdaSection}s for a chosen set of {@link Variable}s. It is used to roll back the state of a {@link BayesianNetwork} in a performant way.
 */
public class LambdaSectionSnapshot {

    private final Map<Variable, LambdaSection> affectedVariablesCache;

    public LambdaSectionSnapshot(List<Vertex> latentVertices) {
        this.affectedVariablesCache = createVariablesAffectedByCache(latentVertices);
    }

    public double logProb(Set<? extends Variable> variables) {
         Set<Vertex> lambdaSectionUnion = new HashSet<>();
        for (Variable v : variables) {
            lambdaSectionUnion.addAll(affectedVariablesCache.get(v).getLatentAndObservedVertices());
        }
        return ProbabilityCalculator.calculateLogProbFor(lambdaSectionUnion);
    }

    public Set<Vertex> getAllVerticesAffectedBy(Set<? extends Variable> variables) {
        Set<Vertex> allAffectedVariables = new HashSet<>();
        for (Variable variable : variables) {
            allAffectedVariables.addAll(affectedVariablesCache.get(variable).getAllVertices());
        }
        return allAffectedVariables;
    }

    /**
     * This creates a cache of potentially all vertices downstream to an observed or probabilistic vertex
     * from each latent vertex.
     *
     * @param latentVertices      The latent vertices to create a cache for
     * @return A variable to Lambda Section map that represents the downstream Lambda Section for each latent vertex.
     * This Lambda Section may include all of the nonprobabilistic vertices if useCacheOnRejection is enabled.
     */
    private static Map<Variable, LambdaSection> createVariablesAffectedByCache(List<Vertex> latentVertices) {
        return latentVertices.stream()
            .collect(Collectors.toMap(
                v -> v,
                v -> LambdaSection.getDownstreamLambdaSection(v, true)
            ));
    }
}
