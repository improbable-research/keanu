package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.algorithms.variational.optimizer.Variable;
import io.improbable.keanu.vertices.Vertex;

import java.util.Set;

public class CascadeOnRejection implements ProposalRejectionStrategy {

    private Set<? extends Variable> variables;

    @Override
    public void prepare(Set<Variable> variables) {
        this.variables = variables;
    }

    @Override
    public void handle() {
        VertexValuePropagation.cascadeUpdate((Set<Vertex>) variables);
    }
}
