package io.improbable.keanu.algorithms.mcmc.proposal;

import io.improbable.keanu.algorithms.Variable;

import java.util.List;
import java.util.Set;

public interface MHStepVariableSelector {
    MHStepVariableSelector SINGLE_VARIABLE_SELECTOR = SingleVariableSelector.INSTANCE;
    MHStepVariableSelector FULL_VARIABLE_SELECTOR = FullVariableSelector.INSTANCE;

    Set<Variable> select(List<? extends Variable> latentVariables, int sampleNumber);
}
