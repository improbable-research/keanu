package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary;

import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class IntegerMultiplicationVertex extends IntegerBinaryOpVertex {

    public IntegerMultiplicationVertex(IntegerVertex a, IntegerVertex b) {
        super(a, b);
    }

    @Override
    protected Integer op(Integer a, Integer b) {
        return a * b;
    }
}
