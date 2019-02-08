package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.algorithms.mcmc.proposal.Proposal;
import io.improbable.keanu.vertices.Vertex;

import java.util.Map;
import java.util.Set;

import static java.util.stream.Collectors.toMap;

public class KeanuProposalApplicationStrategy implements ProposalApplicationStrategy {

    private final Map<VariableReference, Vertex> vertexLookup;

    public KeanuProposalApplicationStrategy(Set<Vertex> vertices) {
        this.vertexLookup = vertices.stream()
            .collect(toMap(Vertex::getReference, v -> v));
    }

    @Override
    public void apply(Proposal proposal) {
        Map<VariableReference, Object> proposalTo = proposal.getProposalTo();
        for (Map.Entry<VariableReference, Object> variableProposal : proposalTo.entrySet()) {
            Vertex vertex = vertexLookup.get(variableProposal.getKey());
            vertex.setValue(variableProposal.getValue());
        }
    }
}
