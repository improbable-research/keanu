package io.improbable.keanu.vertices.intgrtensor.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.intgrtensor.IntegerVertex;

public class IntegerMultiplicationVertex extends IntegerBinaryOpVertex {

    public IntegerMultiplicationVertex(IntegerVertex a, IntegerVertex b) {
        super(a, b);
    }

    @Override
    protected IntegerTensor op(IntegerTensor a, IntegerTensor b) {
        return a.times(b);
    }
}
