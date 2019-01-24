package io.improbable.keanu.algorithms.mcmc.proposal;

import com.google.common.collect.Lists;
import io.improbable.keanu.algorithms.variational.optimizer.Variable;
import io.improbable.keanu.algorithms.variational.optimizer.VariableReference;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import static java.util.stream.Collectors.toMap;

public class Proposal {

    private final Map<Variable, Object> perVariableProposalTo;
    private final Map<Variable, Object> perVariableProposalFrom;
    private final List<ProposalListener> listeners = Lists.newArrayList();

    public Proposal() {
        this.perVariableProposalTo = new HashMap<>();
        this.perVariableProposalFrom = new HashMap<>();
    }

    public <T> void setProposal(Variable<T, ?> variable, T to) {
        perVariableProposalFrom.put(variable, variable.getValue());
        perVariableProposalTo.put(variable, to);
    }

    public void addListener(ProposalListener listener) {
        this.listeners.add(listener);
    }

    public void addListeners(List<ProposalListener> listeners) {
        this.listeners.addAll(listeners);
    }

    public <T> T getProposalTo(Variable<T, ?> variable) {
        return (T) perVariableProposalTo.get(variable);
    }

    public Map<VariableReference, Object> getProposalTo() {
        return perVariableProposalTo.entrySet().stream().collect(toMap(e -> e.getKey().getReference(), Map.Entry::getValue));
    }

    public <T> T getProposalFrom(Variable<T, ?> variable) {
        return (T) perVariableProposalFrom.get(variable);
    }

    public Set<Variable> getVariablesWithProposal() {
        return perVariableProposalTo.keySet();
    }

    public void apply() {
        Set<Variable> variables = perVariableProposalTo.keySet();
        for (Variable v : variables) {
            v.setValue(getProposalTo(v));
        }
        for (ProposalListener listener : listeners) {
            listener.onProposalApplied(this);
        }
    }

    public void reject() {
        Set<Variable> variables = perVariableProposalTo.keySet();
        for (Variable v : variables) {
            v.setValue(getProposalFrom(v));
        }
        for (ProposalListener listener : listeners) {
            listener.onProposalRejected(this);
        }
    }

}
