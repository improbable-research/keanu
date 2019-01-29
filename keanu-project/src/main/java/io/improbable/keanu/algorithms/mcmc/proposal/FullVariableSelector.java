package io.improbable.keanu.algorithms.mcmc.proposal;

import com.google.common.collect.ImmutableSet;
import io.improbable.keanu.algorithms.Variable;

import java.util.List;
import java.util.Set;

public final class FullVariableSelector implements MHStepVariableSelector {

    static final FullVariableSelector INSTANCE = new FullVariableSelector();

    private FullVariableSelector() {
    }

    @Override
    public Set<Variable> select(List<? extends Variable> latentVariables, int sampleNumber) {
        return ImmutableSet.copyOf(latentVariables);
    }
}
