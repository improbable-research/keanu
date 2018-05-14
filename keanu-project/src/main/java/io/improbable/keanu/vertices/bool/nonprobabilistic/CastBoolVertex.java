package io.improbable.keanu.vertices.bool.nonprobabilistic;

import io.improbable.keanu.vertices.Vertex;

import java.util.Random;

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
    public Boolean sample(Random random) {
        return inputVertex.sample(random);
    }

}

