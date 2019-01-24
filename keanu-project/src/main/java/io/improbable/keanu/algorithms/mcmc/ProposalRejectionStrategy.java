package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.mcmc.proposal.Proposal;
import io.improbable.keanu.algorithms.mcmc.proposal.ProposalListener;
import io.improbable.keanu.algorithms.variational.optimizer.Variable;

import java.util.Set;

public interface ProposalRejectionStrategy extends ProposalListener {
}
