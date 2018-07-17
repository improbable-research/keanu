package io.improbable.keanu.algorithms.mcmc.proposals;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import java.util.Set;

public class PriorProposal implements ProposalDistribution {

    public static final PriorProposal SINGLETON = new PriorProposal();

    private PriorProposal() {
    }

    @Override
    public Proposal getProposal(Set<Vertex> vertices, KeanuRandom random) {
        Proposal proposal = new Proposal();
        for (Vertex<?> vertex : vertices) {
            setFor(vertex, random, proposal);
        }
        return proposal;
    }

    private <T> void setFor(Vertex<T> vertex, KeanuRandom random, Proposal proposal) {
        proposal.setProposal(vertex, vertex.getValue(), vertex.sample(random));
    }
}
