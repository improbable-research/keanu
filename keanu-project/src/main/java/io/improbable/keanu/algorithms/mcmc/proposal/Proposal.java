package io.improbable.keanu.algorithms.mcmc.proposal;

import com.google.common.collect.Lists;
import io.improbable.keanu.algorithms.variational.optimizer.Variable;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class Proposal {

    private final Map<Variable, Object> perVertexProposalTo;
    private final Map<Variable, Object> perVertexProposalFrom;
    private final List<ProposalListener> listeners = Lists.newArrayList();

    public Proposal() {
        this.perVertexProposalTo = new HashMap<>();
        this.perVertexProposalFrom = new HashMap<>();
    }

    public <T> void setProposal(Variable<T> vertex, T to) {
        perVertexProposalFrom.put(vertex, vertex.getValue());
        perVertexProposalTo.put(vertex, to);
    }

    public void addListener(ProposalListener listener) {
        this.listeners.add(listener);
    }

    public void addListeners(List<ProposalListener> listeners) {
        this.listeners.addAll(listeners);
    }

    public <T> T getProposalTo(Variable<T> vertex) {
        return (T) perVertexProposalTo.get(vertex);
    }

    public <T> T getProposalFrom(Variable<T> vertex) {
        return (T) perVertexProposalFrom.get(vertex);
    }

    public Set<Variable> getVerticesWithProposal() {
        return perVertexProposalTo.keySet();
    }

    public void apply() {
        Set<Variable> vertices = perVertexProposalTo.keySet();
        for (Variable v : vertices) {
            v.setValue(getProposalTo(v));
        }
        for (ProposalListener listener : listeners) {
            listener.onProposalApplied(this);
        }
    }

    public void reject() {
        Set<Variable> vertices = perVertexProposalTo.keySet();
        for (Variable v : vertices) {
            v.setValue(getProposalFrom(v));
        }
        for (ProposalListener listener : listeners) {
            listener.onProposalRejected(this);
        }
    }

}
