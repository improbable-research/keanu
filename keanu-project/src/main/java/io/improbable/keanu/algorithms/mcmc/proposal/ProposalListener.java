package io.improbable.keanu.algorithms.mcmc.proposal;

public interface ProposalListener {
    void onProposalCreated(Proposal proposal);

    void onProposalRejected(Proposal proposal);

}
