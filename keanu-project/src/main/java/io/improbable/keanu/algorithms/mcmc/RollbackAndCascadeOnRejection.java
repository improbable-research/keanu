package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.algorithms.mcmc.proposal.Proposal;
import io.improbable.keanu.vertices.Vertex;

import java.util.HashMap;
import java.util.Map;

/**
 * When a proposal is created, take a snapshot of the vertices' values.
 * When a proposal is rejected, reset the Bayes Net to those values and cascade update.
 * This is less performant than {@link RollBackToCachedValuesOnRejection}
 */
public class RollbackAndCascadeOnRejection implements ProposalRejectionStrategy {

    private Map<Vertex, Object> fromValues;

    @Override
    public void onProposalCreated(Proposal proposal) {

        fromValues = new HashMap<>();
        for (Variable variable : proposal.getVariablesWithProposal()) {

            if (variable instanceof Vertex) {
                fromValues.put((Vertex) variable, variable.getValue());
            } else {
                throw new IllegalArgumentException(this.getClass().getSimpleName() + " is to only be used with Keanu's Vertex");
            }

        }
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
