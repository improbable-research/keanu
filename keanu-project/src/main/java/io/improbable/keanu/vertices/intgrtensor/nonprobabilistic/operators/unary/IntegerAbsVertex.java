package io.improbable.keanu.vertices.intgrtensor.nonprobabilistic.operators.unary;


import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.intgrtensor.IntegerVertex;

public class IntegerAbsVertex extends IntegerUnaryOpVertex {

    public IntegerAbsVertex(IntegerVertex inputVertex) {
        super(inputVertex.getShape(), inputVertex);
    }

    @Override
    protected IntegerTensor op(IntegerTensor a) {
        return a.abs();
    }
}
