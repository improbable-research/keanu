package io.improbable.keanu.algorithms.mcmc.proposal;

import io.improbable.keanu.algorithms.variational.optimizer.Variable;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import java.util.Set;

public interface ProposalDistribution {

    Proposal getProposal(Set<Variable> variables, KeanuRandom random);

    <T> double logProb(Probabilistic<T> variable, T ofValue, T givenValue);

    /**
     * Represents q(x|x') where q is the proposal distribution,
     * x' is the proposal to value and x is the proposal from value.
     *
     * @param proposal A proposal value for each variable that contains
     *                 a from and a to value.
     * @return the sum of the log probabilities for each variable at x given x'
     */
    default double logProbAtFromGivenTo(Proposal proposal) {
        double sumLogProb = 0.0;
        for (Variable v : proposal.getVariablesWithProposal()) {
            sumLogProb += logProb((Probabilistic) v, proposal.getProposalFrom(v), proposal.getProposalTo(v));
        }
        return sumLogProb;
    }

    /**
     * Represents q(x'|x) where q is the proposal distribution,
     * x' is the Proposal To value and x is the Proposal From value.
     *
     * @param proposal A proposal value for each variable that contains
     *                 a from and a to value.
     * @return the sum of the log probabilities for each variable at x' given x
     */
    default double logProbAtToGivenFrom(Proposal proposal) {
        double sumLogProb = 0.0;
        for (Variable v : proposal.getVariablesWithProposal()) {
            sumLogProb += logProb((Probabilistic) v, proposal.getProposalTo(v), proposal.getProposalFrom(v));
        }
        return sumLogProb;
    }

    void onProposalRejected();
}
