package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.algorithms.mcmc.proposal.Proposal;
import io.improbable.keanu.algorithms.variational.optimizer.Variable;
import io.improbable.keanu.vertices.Vertex;

import java.util.Set;

public class CascadeOnApplication implements ProposalApplicationStrategy {

    @Override
    public void apply(Proposal proposal, Set<? extends Variable> inputs) {
        VertexValuePropagation.cascadeUpdate((Set<Vertex>) inputs);
    }
}
