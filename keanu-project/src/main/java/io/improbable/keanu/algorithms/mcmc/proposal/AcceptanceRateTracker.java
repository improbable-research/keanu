package io.improbable.keanu.algorithms.mcmc.proposal;

import com.google.common.collect.Maps;
import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.VariableReference;

import java.util.Map;

public class AcceptanceRateTracker implements ProposalListener {

    private Map<VariableReference, Counter> numApplied = Maps.newHashMap();
    private Map<VariableReference, Counter> numRejected = Maps.newHashMap();

    @Override
    public void onProposalCreated(Proposal proposal) {
        for (Variable variable : proposal.getVariablesWithProposal()) {
            numApplied.computeIfAbsent(variable.getReference(), i -> new Counter()).increment();
        }
    }

    @Override
    public void onProposalRejected(Proposal proposal) {
        for (Variable variable : proposal.getVariablesWithProposal()) {
            numRejected.computeIfAbsent(variable.getReference(), i -> new Counter()).increment();
        }
    }

    public double getAcceptanceRate(VariableReference variableReference) {
        if (!numApplied.keySet().contains(variableReference)) {
            throw new IllegalStateException("No proposals have been registered for " + variableReference);
        }
        return 1. - (double) numRejected.getOrDefault(variableReference, new Counter()).getValue() / numApplied.get(variableReference).getValue();
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
