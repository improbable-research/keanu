package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary;


import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class IntegerAbsVertex extends IntegerUnaryOpVertex {

    public IntegerAbsVertex(IntegerVertex inputVertex) {
        super(inputVertex);
    }

    @Override
    protected Integer op(Integer a) {
        return Math.abs(a);
    }
}
