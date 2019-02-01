package io.improbable.keanu.algorithms.mcmc.proposal;


import io.improbable.keanu.vertices.RandomVariable;

import java.util.Collections;
import java.util.List;
import java.util.Set;

public final class SingleVariableSelector implements MHStepVariableSelector {

    static final SingleVariableSelector INSTANCE = new SingleVariableSelector();

    private SingleVariableSelector() {
    }

    @Override
    public Set<? extends RandomVariable> select(List<? extends RandomVariable> latentVariables, int sampleNumber) {
        RandomVariable chosenVariable = latentVariables.get(sampleNumber % latentVariables.size());
        return Collections.singleton(chosenVariable);
    }
}
