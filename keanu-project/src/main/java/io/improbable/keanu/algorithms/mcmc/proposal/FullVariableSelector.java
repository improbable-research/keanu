package io.improbable.keanu.algorithms.mcmc.proposal;

import com.google.common.collect.ImmutableSet;
import io.improbable.keanu.vertices.ProbabilisticVariable;

import java.util.List;
import java.util.Set;

public final class FullVariableSelector implements MHStepVariableSelector {

    static final FullVariableSelector INSTANCE = new FullVariableSelector();

    private FullVariableSelector() {
    }

    @Override
    public Set<ProbabilisticVariable> select(List<? extends ProbabilisticVariable> latentVariables, int sampleNumber) {
        return ImmutableSet.copyOf(latentVariables);
    }
}
