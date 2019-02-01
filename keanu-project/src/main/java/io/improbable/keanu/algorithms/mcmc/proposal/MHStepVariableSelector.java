package io.improbable.keanu.algorithms.mcmc.proposal;

import io.improbable.keanu.vertices.RandomVariable;

import java.util.List;
import java.util.Set;

public interface MHStepVariableSelector {
    MHStepVariableSelector SINGLE_VARIABLE_SELECTOR = SingleVariableSelector.INSTANCE;
    MHStepVariableSelector FULL_VARIABLE_SELECTOR = FullVariableSelector.INSTANCE;

    Set<? extends RandomVariable> select(List<? extends RandomVariable> latentVariables, int sampleNumber);
}
