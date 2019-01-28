package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.algorithms.mcmc.proposal.Proposal;
import io.improbable.keanu.algorithms.variational.optimizer.Variable;
import io.improbable.keanu.algorithms.variational.optimizer.VariableReference;
import io.improbable.keanu.vertices.Vertex;

import java.util.Collection;
import java.util.Map;
import java.util.stream.Collectors;

public class RollbackAndCascadeOnRejection implements ProposalRejectionStrategy {

    private Map<Vertex, Object> fromValues;
    private final Map<VariableReference, Vertex> vertexLookup;

    public RollbackAndCascadeOnRejection(Collection<Vertex> vertices) {
        vertexLookup = vertices.stream().collect(Collectors.toMap(Variable::getReference, v -> v));
    }

    @Override
    public void onProposalCreated(Proposal proposal) {
        fromValues = proposal.getVariablesWithProposal().stream()
            .collect(Collectors.toMap(v -> vertexLookup.get(v.getReference()), Variable::getValue));
    }

    @Override
    public void onProposalRejected(Proposal proposal) {

        for (Map.Entry<Vertex, Object> entry : fromValues.entrySet()) {
            Object oldValue = entry.getValue();
            Vertex vertex = entry.getKey();
            vertex.setValue(oldValue);
        }
        VertexValuePropagation.cascadeUpdate(fromValues.keySet());
    }
}
