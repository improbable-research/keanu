package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.variational.optimizer.Variable;

import java.util.Set;

public interface ProposalRejectionStrategy {
    void prepare(Set<Variable> chosenVertices);
    void handle();
}
