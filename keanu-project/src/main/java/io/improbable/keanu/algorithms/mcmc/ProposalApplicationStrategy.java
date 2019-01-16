package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.mcmc.proposal.Proposal;
import io.improbable.keanu.algorithms.variational.optimizer.Variable;

import java.util.Set;

public interface ProposalApplicationStrategy {

    /**
     * Any actions taken after after applying the provided proposal
     * from a sampling algorithm to its variables.
     *
     * No action will be taken in most cases.
     *
     * @param proposal  the proposal that has been applied
     * @param inputs    the variables that have been proposed
     */
    void apply(Proposal proposal, Set<? extends Variable> inputs);

}
