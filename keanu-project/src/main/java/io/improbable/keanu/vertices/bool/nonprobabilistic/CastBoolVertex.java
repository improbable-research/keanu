package io.improbable.keanu.vertices.bool.nonprobabilistic;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class CastBoolVertex extends NonProbabilisticBool {

    private final Vertex<? extends BooleanTensor> inputVertex;

    public CastBoolVertex(Vertex<? extends BooleanTensor> inputVertex) {
        this.inputVertex = inputVertex;
        setParents(inputVertex);
    }

    @Override
    public BooleanTensor getDerivedValue() {
        return inputVertex.getValue();
    }

    @Override
    public BooleanTensor sample(KeanuRandom random) {
        return inputVertex.sample(random);
    }

}

