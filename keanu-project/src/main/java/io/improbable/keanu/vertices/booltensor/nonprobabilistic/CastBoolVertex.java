package io.improbable.keanu.vertices.booltensor.nonprobabilistic;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

public class CastBoolVertex extends NonProbabilisticBool {

    private final Vertex<BooleanTensor> inputVertex;

    public CastBoolVertex(Vertex<BooleanTensor> inputVertex) {
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

