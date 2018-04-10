package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary;

import io.improbable.keanu.vertices.intgr.IntegerVertex;


public class IntegerDivisionVertex extends IntegerBinaryOpVertex {

    public IntegerDivisionVertex(IntegerVertex a, IntegerVertex b) {
        super(a, b);
    }

    protected Integer op(Integer a, Integer b) {
        return a / b;
    }
}
