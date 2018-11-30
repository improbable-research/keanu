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
        if (numApplied.keySet().contains(key)) {
            log.trace(String.format(" Applied proposal: %.4f for %s", getAcceptanceRate(key), proposal.getVerticesWithProposal()));
        } else {
            log.trace(String.format(" Applied first proposal for %s", proposal.getVerticesWithProposal()));
            numApplied.put(key, 0);
        }
        Integer previousValue = numApplied.get(key);
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
