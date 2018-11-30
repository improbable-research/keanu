package io.improbable.keanu.algorithms.mcmc.proposal;

import com.google.common.collect.Maps;
import io.improbable.keanu.vertices.Vertex;

import java.util.Map;
import java.util.Set;

public class AcceptanceRateTracker implements ProposalListener {

    private Map<Set<Vertex>, Integer> numApplied = Maps.newHashMap();
    private Map<Set<Vertex>, Integer> numRejected = Maps.newHashMap();

    @Override
    public void onProposalApplied(Proposal proposal) {
        Set<Vertex> key = proposal.getVerticesWithProposal();
        Integer previousValue = numApplied.getOrDefault(key, 0);
        numApplied.put(key, previousValue + 1);
    }

    public double getAcceptanceRate(Set<Vertex> verticesWithProposal) {
        if (!numApplied.keySet().contains(verticesWithProposal)) {
            throw new IllegalStateException("No proposals have been registered for " + verticesWithProposal);
        }
        return 1. - (double) numRejected.getOrDefault(verticesWithProposal, 0) / numApplied.get(verticesWithProposal);
    }

    @Override
    public void onProposalRejected(Proposal proposal) {
        Set<Vertex> key = proposal.getVerticesWithProposal();
        Integer previousValue = numRejected.getOrDefault(key, 0);
        numRejected.put(key, previousValue + 1);
    }
}
