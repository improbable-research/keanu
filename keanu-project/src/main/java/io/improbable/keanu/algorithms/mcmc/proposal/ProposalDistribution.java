package io.improbable.keanu.algorithms.mcmc.proposal;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import java.util.Set;

public interface ProposalDistribution {

    ProposalDistribution usePrior = PriorProposalDistribution.SINGLETON;

    Proposal getProposal(Set<Vertex> vertices, KeanuRandom random);
}
