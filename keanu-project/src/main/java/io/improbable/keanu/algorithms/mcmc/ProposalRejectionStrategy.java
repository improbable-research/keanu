package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.mcmc.proposal.ProposalListener;


/**
 * What to do when a {@link io.improbable.keanu.algorithms.mcmc.proposal.Proposal} is rejected by {@link MetropolisHastings}.
 * Options are {@link RollBackToCachedValuesOnRejection} and {@link RollbackAndCascadeOnRejection}.
 */
public interface ProposalRejectionStrategy extends ProposalListener {
}
