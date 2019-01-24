package io.improbable.keanu.algorithms.mcmc.proposal;

import java.util.List;

public class ProposalNotifier {
    private final List<ProposalListener> listeners;
    private Proposal proposal;

    public ProposalNotifier(List<ProposalListener> listeners) {
        this.listeners = listeners;
    }

    public void notifyProposalCreated(Proposal proposal) {
        this.proposal = proposal;
        for (ProposalListener listener : listeners) {
            listener.onProposalCreated(proposal);
        }
    }

    public void notifyProposalRejected() {
        for (ProposalListener listener : listeners) {
            listener.onProposalRejected(proposal);
        }
    }
}
