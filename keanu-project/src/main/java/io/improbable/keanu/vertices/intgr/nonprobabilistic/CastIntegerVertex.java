package io.improbable.keanu.vertices.intgr.nonprobabilistic;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class CastIntegerVertex extends IntegerVertex implements NonProbabilistic<IntegerTensor> {

    private final Vertex<IntegerTensor> inputVertex;

    public CastIntegerVertex(Vertex<IntegerTensor> inputVertex) {
        super(inputVertex.getShape());
        this.inputVertex = inputVertex;
        setParents(inputVertex);
    }

    @Override
    public IntegerTensor sample(KeanuRandom random) {
        return inputVertex.sample(random);
    }

    @Override
    public IntegerTensor calculate() {
        return inputVertex.getValue();
    }
}
