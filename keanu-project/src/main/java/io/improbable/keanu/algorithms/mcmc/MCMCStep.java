package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.algorithms.mcmc.proposals.Proposal;
import io.improbable.keanu.algorithms.mcmc.proposals.ProposalDistribution;
import io.improbable.keanu.network.LambdaSection;
import io.improbable.keanu.network.NetworkSnapshot;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public class MCMCStep {

    public static final double LOG_ZERO_PROBABILITY = Double.NEGATIVE_INFINITY;

    private final ProposalDistribution proposalDistribution;
    private final boolean useCacheOnRejection;
    private final Map<Vertex, LambdaSection> affectedVerticesCache;

    MCMCStep(List<? extends Vertex> latentVertices, ProposalDistribution proposalDistribution, boolean useCacheOnRejection) {
        this.proposalDistribution = proposalDistribution;
        this.useCacheOnRejection = useCacheOnRejection;
        this.affectedVerticesCache = getVerticesAffectedByLatents(
            latentVertices,
            useCacheOnRejection
        );
    }


    public double nextSample(final Set<Vertex> chosenVertices,
                             final double totalLogProbOld,
                             final KeanuRandom random) {
        return nextSample(chosenVertices, totalLogProbOld, 1.0, random);
    }

    /**
     * @param chosenVertices  vertices to get a proposed change for
     * @param totalLogProbOld The log of the previous state's probability
     * @param T               Temperature for simulated annealing. This
     *                        should be constant if no annealing is wanted
     * @param random          source of randomness
     * @return the log probability of the network after either accepting or rejecting the sample
     */
    public double nextSample(final Set<Vertex> chosenVertices,
                             final double totalLogProbOld,
                             final double T,
                             final KeanuRandom random) {

        final double affectedVerticesLogProbOld = sumLogProbabilityOfAffected(chosenVertices, affectedVerticesCache);

        NetworkSnapshot preProposalSnapshot = null;
        if (useCacheOnRejection) {
            preProposalSnapshot = getSnapshotOfAllAffectedVertices(chosenVertices, affectedVerticesCache);
        }

        Proposal proposal = proposalDistribution.getProposal(chosenVertices, random);
        proposal.apply();
        VertexValuePropagation.cascadeUpdate(chosenVertices);

        final double affectedVerticesLogProbNew = sumLogProbabilityOfAffected(chosenVertices, affectedVerticesCache);

        if (affectedVerticesLogProbNew != LOG_ZERO_PROBABILITY) {

            final double totalLogProbNew = totalLogProbOld - affectedVerticesLogProbOld + affectedVerticesLogProbNew;

            final double pqxOld = proposal.logProbAtProposalFrom();
            final double pqxNew = proposal.logProbAtProposalTo();

            final double logR = (totalLogProbNew * (1.0 / T) + pqxOld) - (totalLogProbOld * (1.0 / T) + pqxNew);
            final double r = Math.exp(logR);

            final boolean shouldAccept = r >= random.nextDouble();

            if (shouldAccept) {
                return totalLogProbNew;
            }
        }

        proposal.reject();

        if (useCacheOnRejection) {
            preProposalSnapshot.apply();
        } else {
            VertexValuePropagation.cascadeUpdate(chosenVertices);
        }

        return totalLogProbOld;
    }

    private static NetworkSnapshot getSnapshotOfAllAffectedVertices(final Set<Vertex> chosenVertices,
                                                                    final Map<Vertex, LambdaSection> affectedVertices) {

        Set<Vertex> allAffectedVertices = new HashSet<>();
        for (Vertex vertex : chosenVertices) {
            allAffectedVertices.addAll(affectedVertices.get(vertex).getAllVertices());
        }

        return NetworkSnapshot.create(allAffectedVertices);
    }

    private static double sumLogProbabilityOfAffected(Set<Vertex> vertices,
                                                      Map<Vertex, LambdaSection> affectedVertices) {
        double sumLogProb = 0.0;
        for (Vertex v : vertices) {
            sumLogProb += sumLogProbability(affectedVertices.get(v).getProbabilisticVertices());
        }
        return sumLogProb;
    }

    /**
     * This returns the log probability of typically a subset of a bayesian network.
     *
     * @param vertices Vertices to consider in log probability calculation
     * @return the log probability of the set of vertices
     */
    private static double sumLogProbability(Set<Vertex> vertices) {
        double sumLogProb = 0.0;
        for (Vertex v : vertices) {
            sumLogProb += v.logProbAtValue();
        }
        return sumLogProb;
    }

    private static Map<Vertex, LambdaSection> getVerticesAffectedByLatents(List<? extends Vertex> latentVertices,
                                                                           boolean includeNonProbabilistic) {
        return latentVertices.stream()
            .collect(Collectors.toMap(
                v -> v,
                v -> LambdaSection.getDownstreamLambdaSection(v, includeNonProbabilistic)
            ));
    }

}
