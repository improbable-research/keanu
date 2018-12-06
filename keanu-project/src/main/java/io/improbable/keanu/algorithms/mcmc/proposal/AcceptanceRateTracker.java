package io.improbable.keanu.algorithms.mcmc.proposal;

import com.google.common.collect.Maps;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;

import java.util.Map;

public class AcceptanceRateTracker implements ProposalListener {

    private Map<VertexId, Integer> numApplied = Maps.newHashMap();
    private Map<VertexId, Integer> numRejected = Maps.newHashMap();

    @Override
    public void onProposalApplied(Proposal proposal) {
        for (Vertex vertex : proposal.getVerticesWithProposal()) {
            Integer previousValue = numApplied.getOrDefault(vertex.getId(), 0);
            numApplied.put(vertex.getId(), previousValue + 1);
        }
    }

    public double getAcceptanceRate(VertexId vertexId) {
        if (!numApplied.keySet().contains(vertexId)) {
            throw new IllegalStateException("No proposals have been registered for " + vertexId);
        }
        return 1. - (double) numRejected.getOrDefault(vertexId, 0) / numApplied.get(vertexId);
    }

    @Override
    public void onProposalRejected(Proposal proposal) {
        for (Vertex vertex : proposal.getVerticesWithProposal()) {
            Integer previousValue = numRejected.getOrDefault(vertex.getId(), 0);
            numRejected.put(vertex.getId(), previousValue + 1);
        }
    }
}
