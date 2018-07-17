package io.improbable.keanu.algorithms.mcmc.proposals;

import io.improbable.keanu.vertices.Vertex;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

public class Proposal {

    private final Map<Vertex, Object> perVertexProposalTo;
    private final Map<Vertex, Object> perVertexProposalFrom;

    public Proposal() {
        this.perVertexProposalTo = new HashMap<>();
        this.perVertexProposalFrom = new HashMap<>();
    }

    public <T> void setProposal(Vertex<T> vertex, T from, T to) {
        perVertexProposalFrom.put(vertex, from);
        perVertexProposalTo.put(vertex, to);
    }

    public <T> T getProposalTo(Vertex<T> vertex) {
        return (T) perVertexProposalTo.get(vertex);
    }

    public <T> T getProposalFrom(Vertex<T> vertex) {
        return (T) perVertexProposalFrom.get(vertex);
    }

    public void apply() {
        Set<Vertex> vertices = perVertexProposalTo.keySet();
        for (Vertex v : vertices) {
            v.setValue(getProposalTo(v));
        }
    }

    public void reject() {
        Set<Vertex> vertices = perVertexProposalTo.keySet();
        for (Vertex v : vertices) {
            v.setValue(getProposalFrom(v));
        }
    }

    public double logProbAtProposalFrom() {
        double sumLogProb = 0.0;
        for (Vertex v : perVertexProposalTo.keySet()) {
            sumLogProb += v.logProb(getProposalFrom(v));
        }
        return sumLogProb;
    }

    public double logProbAtProposalTo() {
        double sumLogProb = 0.0;
        for (Vertex v : perVertexProposalTo.keySet()) {
            sumLogProb += v.logProb(getProposalTo(v));
        }
        return sumLogProb;
    }

}
