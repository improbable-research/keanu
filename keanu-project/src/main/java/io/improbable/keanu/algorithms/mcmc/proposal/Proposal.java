package io.improbable.keanu.algorithms.mcmc.proposal;

import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.VariableReference;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

public class Proposal {

    private final Map<Variable, Object> perVariableProposalTo;
    private final Map<Variable, Object> perVariableProposalFrom;

    public Proposal() {
        this.perVariableProposalTo = new HashMap<>();
        this.perVariableProposalFrom = new HashMap<>();
    }

    public <T> void setProposal(Variable<T, ?> variable, T to) {
        perVariableProposalFrom.put(variable, variable.getValue());
        perVariableProposalTo.put(variable, to);
    }

    public <T> T getProposalTo(Variable<T, ?> variable) {
        return (T) perVariableProposalTo.get(variable);
    }

    public Map<VariableReference, Object> getProposalTo() {
        Map<VariableReference, Object> asMap = new HashMap<>();

        for (Map.Entry<Variable, Object> entry : perVariableProposalTo.entrySet()) {
            asMap.put(entry.getKey().getReference(), entry.getValue());
        }

        return asMap;
    }

    public <T> T getProposalFrom(Variable<T, ?> variable) {
        return (T) perVariableProposalFrom.get(variable);
    }

    public Set<Variable> getVariablesWithProposal() {
        return perVariableProposalTo.keySet();
    }
}
