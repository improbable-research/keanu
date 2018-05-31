package io.improbable.keanu.vertices.intgrtensor.nonprobabilistic;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

public class CastIntegerVertex extends NonProbabilisticInteger {

    private final Vertex<IntegerTensor> inputVertex;

    public CastIntegerVertex(Vertex<IntegerTensor> inputVertex) {
        this.inputVertex = inputVertex;
        setParents(inputVertex);
        setValue(IntegerTensor.placeHolder(inputVertex.getShape()));
    }

    @Override
    public IntegerTensor sample(KeanuRandom random) {
        return inputVertex.sample(random);
    }

    @Override
    public IntegerTensor getDerivedValue() {
        return inputVertex.getValue();
    }
}
