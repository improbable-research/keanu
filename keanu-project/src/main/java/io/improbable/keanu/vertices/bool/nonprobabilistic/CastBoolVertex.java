package io.improbable.keanu.vertices.bool.nonprobabilistic;

import io.improbable.keanu.vertices.Vertex;

import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

public class CastBoolVertex extends NonProbabilisticBool {

    private final Vertex<Boolean> inputVertex;

    public CastBoolVertex(Vertex<Boolean> inputVertex) {
        this.inputVertex = inputVertex;
        setParents(inputVertex);
    }

    @Override
    public Boolean getDerivedValue() {
        return inputVertex.getValue();
    }

    @Override
    public Boolean sample(KeanuRandom random) {
        return inputVertex.sample(random);
    }

}

