package io.improbable.keanu.algorithms.mcmc.proposal;

import com.google.common.collect.Maps;
import io.improbable.keanu.vertices.Vertex;
import lombok.extern.slf4j.Slf4j;

import java.util.Map;
import java.util.Set;

@Slf4j
public class AcceptanceRateTracker implements ProposalListener {

    private Map<Set<Vertex>, Integer> numApplied = Maps.newHashMap();
    private Map<Set<Vertex>, Integer> numRejected = Maps.newHashMap();

    @Override
    public void onProposalApplied(Proposal proposal) {
        Set<Vertex> key = proposal.getVerticesWithProposal();
        log.trace(String.format(" Applied proposal: %.4f", getAcceptanceRate(key)));
        Integer previousValue = numApplied.getOrDefault(key, 0);
        numApplied.put(key, previousValue + 1);
    }

    public double getAcceptanceRate(Set<Vertex> verticesWithProposal) {
        return 1. - (double) numRejected.getOrDefault(verticesWithProposal, 0) / numApplied.getOrDefault(verticesWithProposal, 0);
    }

    @Override
    public void onProposalRejected(Proposal proposal) {
        Set<Vertex> key = proposal.getVerticesWithProposal();
        Integer previousValue = numRejected.getOrDefault(key, 0);
        numRejected.put(key, previousValue + 1);
    }
}
