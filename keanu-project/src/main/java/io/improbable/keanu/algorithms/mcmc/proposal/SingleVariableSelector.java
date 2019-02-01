package io.improbable.keanu.algorithms.mcmc.proposal;


import io.improbable.keanu.vertices.ProbabilisticVariable;

import java.util.Collections;
import java.util.List;
import java.util.Set;

public final class SingleVariableSelector implements MHStepVariableSelector {

    static final SingleVariableSelector INSTANCE = new SingleVariableSelector();

    private SingleVariableSelector() {
    }

    @Override
    public Set<ProbabilisticVariable> select(List<? extends ProbabilisticVariable> latentVariables, int sampleNumber) {
        ProbabilisticVariable chosenVariable = latentVariables.get(sampleNumber % latentVariables.size());
        return Collections.singleton(chosenVariable);
    }
}
