package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.mcmc.proposal.Proposal;
import io.improbable.keanu.algorithms.variational.optimizer.Variable;

import java.util.Set;

public class NoActionOnApplication implements ProposalApplicationStrategy {

    @Override
    public void apply(Proposal proposal, Set<? extends Variable> inputs) {
        // no action
    }
}
