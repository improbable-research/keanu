package io.improbable.keanu.algorithms.mcmc.proposal;

import com.google.common.base.Preconditions;
import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;

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
        checkAllVerticesAreProbabilistic(vertices);
        vertexLookup = vertices.stream().collect(Collectors.toMap(Vertex::getReference, v -> v));
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
        proposal.setProposal(variable, (((Probabilistic<T>) vertex).sample(random)));
    }

    @Override
    public void onProposalRejected() {
        proposalNotifier.notifyProposalRejected();
    }

    private void checkAllVerticesAreProbabilistic(Collection<Vertex> vertices) {
        for (Vertex v : vertices) {
            Preconditions.checkArgument(v instanceof Probabilistic, "Prior proposal vertices must be probabilistic. Vertex is: " + v);
        }
    }
}
