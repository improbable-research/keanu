package io.improbable.keanu.algorithms.mcmc.proposal;


import io.improbable.keanu.algorithms.variational.optimizer.Variable;

import java.util.Collections;
import java.util.List;
import java.util.Set;

public final class SingleVariableSelector implements MHStepVariableSelector {

    static final SingleVariableSelector INSTANCE = new SingleVariableSelector();

    private SingleVariableSelector() {
    }

    @Override
    public Set<Variable> select(List<? extends Variable> latentVertices, int sampleNumber) {
        Variable chosenVertex = latentVertices.get(sampleNumber % latentVertices.size());
        return Collections.singleton(chosenVertex);
    }
}
