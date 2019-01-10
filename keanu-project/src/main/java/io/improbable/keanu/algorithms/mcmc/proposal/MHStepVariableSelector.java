package io.improbable.keanu.algorithms.mcmc.proposal;

import io.improbable.keanu.algorithms.variational.optimizer.Variable;

import java.util.List;
import java.util.Set;

public interface MHStepVariableSelector {
    MHStepVariableSelector SINGLE_VARIABLE_SELECTOR = SingleVariableSelector.INSTANCE;
    MHStepVariableSelector FULL_VARIABLE_SELECTOR = FullVariableSelector.INSTANCE;

    Set<? extends Variable> select(List<? extends Variable> vertices, int sampleNumber);
}
