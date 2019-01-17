package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.variational.optimizer.Variable;

import java.util.Set;

public interface ProposalRejectionStrategy {

    /**
     * Action to be taken prior to creating the proposal
     *
     * @param chosenVariables the variables that are going to have new values proposed for
     */
    void prepare(Set<Variable> chosenVariables);

    /**
     * Action to be taken once the proposal has been rejected by a sampling algorithm
     */
    void handle();
}
