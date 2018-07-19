package io.improbable.keanu.algorithms.mcmc.proposal;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import java.util.Set;

public class PriorProposalDistribution implements ProposalDistribution {

    @Override
    public Proposal getProposal(Set<Vertex> vertices, KeanuRandom random) {
        Proposal proposal = new Proposal();
        for (Vertex<?> vertex : vertices) {
            setFor(vertex, random, proposal);
        }
        return proposal;
    }

    private <T> void setFor(Vertex<T> vertex, KeanuRandom random, Proposal proposal) {
        proposal.setProposal(vertex, vertex.sample(random));
    }

    @Override
    public <T> double logProb(Vertex<T> vertex, T ofValue, T givenValue) {
        return vertex.logProb(ofValue);
    }
}
