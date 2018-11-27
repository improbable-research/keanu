package io.improbable.keanu.algorithms.mcmc.proposal;

import com.google.common.collect.ImmutableSet;
import io.improbable.keanu.vertices.Vertex;

import java.util.List;
import java.util.Set;

public final class FullVariableSelector implements MHStepVariableSelector {

    static final FullVariableSelector INSTANCE = new FullVariableSelector();

    private FullVariableSelector() {
    }

    @Override
    public Set<Vertex> select(List<? extends Vertex> latentVertices, int sampleNumber) {
        return ImmutableSet.copyOf(latentVertices);
    }
}
