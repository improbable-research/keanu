package io.improbable.keanu.algorithms.mcmc.proposal;

public interface ProposalListener {
    void onProposalApplied(Proposal proposal);

    void onProposalRejected(Proposal proposal);

}
