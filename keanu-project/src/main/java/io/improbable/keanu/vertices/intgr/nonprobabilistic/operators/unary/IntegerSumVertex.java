package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class IntegerSumVertex extends IntegerUnaryOpVertex {

    public IntegerSumVertex(IntegerVertex inputVertex) {
        super(inputVertex.getShape(), inputVertex);
    }

    @Override
    protected IntegerTensor op(IntegerTensor a) {
        return IntegerTensor.scalar(a.sum());
    }
}
