package io.improbable.keanu.algorithms.mcmc.proposal;

import io.improbable.keanu.vertices.Vertex;
import java.util.List;
import java.util.Set;

public interface MHStepVariableSelector {
    MHStepVariableSelector SINGLE_VARIABLE_SELECTOR = SingleVariableSelector.INSTANCE;

    Set<Vertex> select(List<? extends Vertex> vertices, int sampleNumber);
}
