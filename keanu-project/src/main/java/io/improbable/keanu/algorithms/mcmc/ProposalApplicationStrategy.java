package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.mcmc.proposal.Proposal;

public interface ProposalApplicationStrategy {

    void apply(Proposal proposal);

}
