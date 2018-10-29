package io.improbable.keanu.algorithms.mcmc.proposal;

import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import java.util.Set;

public interface ProposalDistribution {

    static ProposalDistribution usePrior() {
        return new PriorProposalDistribution();
    }

    Proposal getProposal(Set<Vertex> vertices, KeanuRandom random);

    <T> double logProb(Probabilistic<T> vertex, T ofValue, T givenValue);

    /**
     * Represents q(x|x') where q is the proposal distribution,
     * x' is the proposal to value and x is the proposal from value.
     *
     * @param proposal A proposal value for each vertex that contains
     *                 a from and a to value.
     * @return the sum of the log probabilities for each vertex at x given x'
     */
    default double logProbAtFromGivenTo(Proposal proposal) {
        double sumLogProb = 0.0;
        for (Vertex v : proposal.getVerticesWithProposal()) {
            sumLogProb += logProb((Probabilistic) v, proposal.getProposalFrom(v), proposal.getProposalTo(v));
        }
        return sumLogProb;
    }

    /**
     * Represents q(x'|x) where q is the proposal distribution,
     * x' is the Proposal To value and x is the Proposal From value.
     *
     * @param proposal A proposal value for each vertex that contains
     *                 a from and a to value.
     * @return the sum of the log probabilities for each vertex at x' given x
     */
    default double logProbAtToGivenFrom(Proposal proposal) {
        double sumLogProb = 0.0;
        for (Vertex v : proposal.getVerticesWithProposal()) {
            sumLogProb += logProb((Probabilistic) v, proposal.getProposalTo(v), proposal.getProposalFrom(v));
        }
        return sumLogProb;
    }

}
