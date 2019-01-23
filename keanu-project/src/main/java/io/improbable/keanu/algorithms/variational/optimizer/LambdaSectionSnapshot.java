package io.improbable.keanu.algorithms.variational.optimizer;

import io.improbable.keanu.network.LambdaSection;
import io.improbable.keanu.vertices.ProbabilityCalculator;
import io.improbable.keanu.vertices.Vertex;
import lombok.Getter;

import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public class LambdaSectionSnapshot {

    private final List<Vertex> latentVertices;
    @Getter(lazy=true)
    private final Map<Variable, LambdaSection> affectedVariablesCache = update();

    public LambdaSectionSnapshot(List<Vertex> latentVertices) {
        this.latentVertices = latentVertices;
    }

    private Map<Variable, LambdaSection> update() {
        return createVariablesAffectedByCache(latentVertices);
    }

    public double logProb(Set<? extends Variable> variables) {
        update();
        double sumLogProb = 0.0;
        for (Variable v : variables) {
            sumLogProb += ProbabilityCalculator.calculateLogProbFor(this.getAffectedVariablesCache().get(v).getLatentAndObservedVertices());
        }
        return sumLogProb;
    }

    public Set<? extends Variable> getAllVariablesAffectedBy(Set<? extends Variable> variables) {
        Set<Variable> allAffectedVariables = new HashSet<>();
        for (Variable variable : variables) {
            allAffectedVariables.addAll(this.getAffectedVariablesCache().get(variable).getAllVertices());
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
