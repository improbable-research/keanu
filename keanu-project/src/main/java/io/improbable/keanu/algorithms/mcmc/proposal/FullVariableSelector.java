package io.improbable.keanu.algorithms.mcmc.proposal;

import com.google.common.collect.ImmutableSet;
import io.improbable.keanu.algorithms.variational.optimizer.Variable;

import java.util.List;
import java.util.Set;

public final class FullVariableSelector implements MHStepVariableSelector {

    static final FullVariableSelector INSTANCE = new FullVariableSelector();

    private FullVariableSelector() {
    }

    @Override
    public Set<? extends Variable> select(List<? extends Variable> latentVertices, int sampleNumber) {
        return ImmutableSet.copyOf(latentVertices);
    }
}
