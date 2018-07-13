package io.improbable.keanu.algorithms.mcmc.proposal;

import java.util.Collections;
import java.util.List;
import java.util.Set;

import io.improbable.keanu.vertices.Vertex;

public final class SingleVariableSelector implements MHStepVariableSelector {

    static final SingleVariableSelector INSTANCE = new SingleVariableSelector();

    private SingleVariableSelector() {
    }

    @Override
    public Set<Vertex<?>> select(List<? extends Vertex> latentVertices, int sampleNumber) {
        Vertex chosenVertex = latentVertices.get(sampleNumber % latentVertices.size());
        return Collections.singleton(chosenVertex);
    }
}
