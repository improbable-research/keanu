package io.improbable.keanu.network;

import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.vertices.ProbabilityCalculator;
import io.improbable.keanu.vertices.Vertex;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * A snapshot of {@link LambdaSection}s for a chosen set of {@link Variable}s. It is used to roll back the state of a {@link BayesianNetwork} in a performant way.
 */
public class LambdaSectionSnapshot {

    private final Map<Vertex, LambdaSection> affectedVariablesCache;

    public LambdaSectionSnapshot() {
        this.affectedVariablesCache = new HashMap<>();
    }

    public double logProb(Set<? extends Variable> variables) {
        Set<Vertex> lambdaSectionUnion = getAllVerticesAffectedBy(variables);
        return ProbabilityCalculator.calculateLogProbFor(lambdaSectionUnion);
    }

    public Set<Vertex> getAllVerticesAffectedBy(Set<? extends Variable> variables) {

        Set<Vertex> allAffectedVariables = new HashSet<>();
        for (Variable variable : variables) {
            if (variable instanceof Vertex) {

                LambdaSection lambdaSection = createVariablesAffectedByCache((Vertex) variable);

                allAffectedVariables.addAll(lambdaSection.getAllVertices());
            } else {
                throw new IllegalArgumentException(this.getClass().getSimpleName() + " is to only be used with Keanu's Vertex");
            }

        }
        return allAffectedVariables;
    }

    /**
     * This creates a cache of potentially all vertices downstream to an observed or probabilistic vertex
     * from each latent vertex.
     *
     * @param latent The latent vertex to create a cache for
     * @return A variable to Lambda Section map that represents the downstream Lambda Section for the latent vertex.
     * This Lambda Section may include all of the nonprobabilistic vertices if useCacheOnRejection is enabled.
     */
    private LambdaSection createVariablesAffectedByCache(Vertex latent) {
        return affectedVariablesCache.computeIfAbsent(
            latent,
            v -> LambdaSection.getDownstreamLambdaSection(v, true)
        );
    }
}
