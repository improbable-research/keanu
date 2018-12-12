package io.improbable.keanu.algorithms.mcmc.proposal;

import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import java.util.Collections;
import java.util.List;
import java.util.Set;

public class PriorProposalDistribution implements ProposalDistribution {
    private final List<ProposalListener> listeners;

    public PriorProposalDistribution() {
        this(Collections.emptyList());
    }

    public PriorProposalDistribution(List<ProposalListener> listeners) {
        this.listeners = listeners;
    }

    @Override
    public Proposal getProposal(Set<Vertex> vertices, KeanuRandom random) {
        Proposal proposal = new Proposal();
        proposal.addListeners(listeners);
        for (Vertex<?> vertex : vertices) {
            setFor(vertex, random, proposal);
        }
        return proposal;
    }

    @Override
    public <T> double logProb(Probabilistic<T> vertex, T ofValue, T givenValue) {
        return vertex.logProb(ofValue);
    }

    private <T> void setFor(Vertex<T> vertex, KeanuRandom random, Proposal proposal) {
        proposal.setProposal(vertex, vertex.sample(random));
    }

}
