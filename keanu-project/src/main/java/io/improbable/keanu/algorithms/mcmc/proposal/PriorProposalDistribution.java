package io.improbable.keanu.algorithms.mcmc.proposal;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;

import java.util.Collections;
import java.util.List;
import java.util.Set;

public class PriorProposalDistribution implements ProposalDistribution {
    private final ProposalNotifier proposalNotifier;

    public PriorProposalDistribution() {
        this(Collections.emptyList());
    }

    public PriorProposalDistribution(List<ProposalListener> listeners) {
        this.proposalNotifier = new ProposalNotifier(listeners);
    }

    @Override
    public Proposal getProposal(Set<? extends Variable> variables, KeanuRandom random) {
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

        if (variable instanceof Vertex && variable instanceof Probabilistic) {
            Vertex<T, ?> vertex = (Vertex<T, ?>) variable;
            proposal.setProposal(variable, ((Probabilistic<T>) vertex).sample(random));
        } else {
            throw new IllegalArgumentException(this.getClass().getSimpleName() + " is to only be used with Keanu's Vertex");
        }
    }

    @Override
    public void onProposalRejected() {
        proposalNotifier.notifyProposalRejected();
    }

}
