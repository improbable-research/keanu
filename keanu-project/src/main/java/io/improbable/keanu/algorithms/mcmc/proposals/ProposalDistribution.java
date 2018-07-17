package io.improbable.keanu.algorithms.mcmc.proposals;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import java.util.Set;

public interface ProposalDistribution {

    Proposal getProposal(Set<Vertex> vertices, KeanuRandom random);
}
