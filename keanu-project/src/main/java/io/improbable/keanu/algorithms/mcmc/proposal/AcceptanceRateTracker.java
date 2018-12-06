package io.improbable.keanu.algorithms.mcmc.proposal;

import com.google.common.collect.Maps;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;

import java.util.Map;

public class AcceptanceRateTracker implements ProposalListener {

    private Map<VertexId, Counter> numApplied = Maps.newHashMap();
    private Map<VertexId, Counter> numRejected = Maps.newHashMap();

    @Override
    public void onProposalApplied(Proposal proposal) {
        for (Vertex vertex : proposal.getVerticesWithProposal()) {
            numApplied.computeIfAbsent(vertex.getId(), i -> new Counter()).increment();
        }
    }

    public double getAcceptanceRate(VertexId vertexId) {
        if (!numApplied.keySet().contains(vertexId)) {
            throw new IllegalStateException("No proposals have been registered for " + vertexId);
        }
        return 1. - (double) numRejected.getOrDefault(vertexId, new Counter()).getValue() / numApplied.get(vertexId).getValue();
    }

    @Override
    public void onProposalRejected(Proposal proposal) {
        for (Vertex vertex : proposal.getVerticesWithProposal()) {
            numRejected.computeIfAbsent(vertex.getId(), i -> new Counter()).increment();
        }
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
