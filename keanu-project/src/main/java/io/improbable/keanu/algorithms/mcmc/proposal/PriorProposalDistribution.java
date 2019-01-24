package io.improbable.keanu.algorithms.mcmc.proposal;

import io.improbable.keanu.algorithms.variational.optimizer.Variable;
import io.improbable.keanu.algorithms.variational.optimizer.VariableReference;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public class PriorProposalDistribution implements ProposalDistribution {
    private final Map<VariableReference, Vertex> vertexLookup;
    private final ProposalNotifier proposalNotifier;

    public PriorProposalDistribution(Collection<Vertex> vertices) {
        this(vertices, Collections.emptyList());
    }

    public PriorProposalDistribution(Collection<Vertex> vertices, List<ProposalListener> listeners) {
        vertexLookup = vertices.stream().collect(Collectors.toMap(v -> v.getReference(), v -> v));
        this.proposalNotifier = new ProposalNotifier(listeners);

    }

    @Override
    public Proposal getProposal(Set<Variable> variables, KeanuRandom random) {
        Proposal proposal = new Proposal();
        for (Variable<?, ?> variable : variables) {
            setFor(variable, random, proposal);
        }
        proposalNotifier.notifyProposalCreated(proposal);
        return proposal;
    }

    @Override
    public <T> double logProb(Probabilistic<T> variable, T ofValue, T givenValue) {
        return variable.logProb(ofValue);
    }

    private <T> void setFor(Variable<T, ?> variable, KeanuRandom random, Proposal proposal) {
        Vertex<T> vertex = vertexLookup.get(variable.getReference());
        proposal.setProposal(variable, vertex.sample(random));
    }

    @Override
    public void onProposalRejected() {
        proposalNotifier.notifyProposalRejected();
    }
}
