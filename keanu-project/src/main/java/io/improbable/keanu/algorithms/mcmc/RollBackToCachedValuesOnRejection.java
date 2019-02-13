package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.mcmc.proposal.Proposal;
import io.improbable.keanu.network.LambdaSectionSnapshot;
import io.improbable.keanu.network.NetworkSnapshot;
import io.improbable.keanu.vertices.Vertex;

import java.util.Set;

/**
 * When a proposal is created, take a snapshot of the vertices' {@link io.improbable.keanu.network.LambdaSection}s
 * When a proposal is rejected, apply the snapshot to reset the Bayes Net to the old values.
 * This is more performant than {@link RollbackAndCascadeOnRejection}
 */
public class RollBackToCachedValuesOnRejection implements ProposalRejectionStrategy {
    private final LambdaSectionSnapshot lambdaSectionSnapshot;
    private NetworkSnapshot networkSnapshot;

    public RollBackToCachedValuesOnRejection() {
        lambdaSectionSnapshot = new LambdaSectionSnapshot();
    }

    @Override
    public void onProposalCreated(Proposal proposal) {
        Set<Vertex> affectedVariables = lambdaSectionSnapshot.getAllVerticesAffectedBy(proposal.getVariablesWithProposal());
        networkSnapshot = NetworkSnapshot.create(affectedVariables);
    }

    @Override
    public void onProposalRejected(Proposal proposal) {
        networkSnapshot.apply();
    }
}
