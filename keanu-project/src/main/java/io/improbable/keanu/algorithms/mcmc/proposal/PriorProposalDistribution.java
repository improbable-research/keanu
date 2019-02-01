package io.improbable.keanu.algorithms.mcmc.proposal;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.RandomVariable;

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
    public Proposal getProposal(Set<? extends RandomVariable> variables, KeanuRandom random) {
        Proposal proposal = new Proposal();
        for (RandomVariable<?, ?> variable : variables) {
            setFor(variable, random, proposal);
        }
        proposalNotifier.notifyProposalCreated(proposal);
        return proposal;
    }

    @Override
    public <T> double logProb(Probabilistic<T> variable, T ofValue, T givenValue) {
        return variable.logProb(ofValue);
    }

    private <T> void setFor(RandomVariable<T, ?> variable, KeanuRandom random, Proposal proposal) {
        proposal.setProposal(variable, variable.sample(random));
    }

    @Override
    public void onProposalRejected() {
        proposalNotifier.notifyProposalRejected();
    }

}
