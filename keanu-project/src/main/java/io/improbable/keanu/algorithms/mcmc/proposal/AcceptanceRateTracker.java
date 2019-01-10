package io.improbable.keanu.algorithms.mcmc.proposal;

import com.google.common.collect.Maps;
import io.improbable.keanu.algorithms.variational.optimizer.Variable;
import io.improbable.keanu.algorithms.variational.optimizer.VariableReference;

import java.util.Map;

public class AcceptanceRateTracker implements ProposalListener {

    private Map<VariableReference, Counter> numApplied = Maps.newHashMap();
    private Map<VariableReference, Counter> numRejected = Maps.newHashMap();

    @Override
    public void onProposalApplied(Proposal proposal) {
        for (Variable vertex : proposal.getVerticesWithProposal()) {
            numApplied.computeIfAbsent(vertex.getReference(), i -> new Counter()).increment();
        }
    }

    @Override
    public void onProposalRejected(Proposal proposal) {
        for (Variable vertex : proposal.getVerticesWithProposal()) {
            numRejected.computeIfAbsent(vertex.getReference(), i -> new Counter()).increment();
        }
    }

    public double getAcceptanceRate(VariableReference vertexId) {
        if (!numApplied.keySet().contains(vertexId)) {
            throw new IllegalStateException("No proposals have been registered for " + vertexId);
        }
        return 1. - (double) numRejected.getOrDefault(vertexId, new Counter()).getValue() / numApplied.get(vertexId).getValue();
    }

    private class Counter {
        private int count;

        int getValue() {
            return count;
        }

        int increment() {
            return count++;
        }
    }
}
