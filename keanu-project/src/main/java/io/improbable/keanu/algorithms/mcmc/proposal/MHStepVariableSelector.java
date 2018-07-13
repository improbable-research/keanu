package io.improbable.keanu.algorithms.mcmc.proposal;

import java.util.List;
import java.util.Set;

import io.improbable.keanu.vertices.Vertex;

public interface MHStepVariableSelector {
    MHStepVariableSelector SINGLE_VARIABLE_SELECTOR = SingleVariableSelector.INSTANCE;

    Set<Vertex<?>> select(List<? extends Vertex> vertices, int sampleNumber);
}
