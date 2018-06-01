package io.improbable.keanu.vertices.intgrtensor.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.intgrtensor.IntegerVertex;

public class IntegerDifferenceVertex extends IntegerBinaryOpVertex {

    public IntegerDifferenceVertex(IntegerVertex a, IntegerVertex b) {
        super(a, b);
    }

    protected IntegerTensor op(IntegerTensor a, IntegerTensor b) {
        return a.minus(b);
    }
}
