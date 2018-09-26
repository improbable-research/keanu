package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class IntegerMaxVertex extends IntegerBinaryOpVertex {

    /**
     * Finds the maximum between two vertices
     *
     * @param left  one of the vertices to find the maximum of
     * @param right one of the vertices to find the maximum of
     */
    public IntegerMaxVertex(IntegerVertex left, IntegerVertex right) {
        super(left, right);
    }

    @Override
    protected IntegerTensor op(IntegerTensor l, IntegerTensor r) {
        return l.max(r);
    }
}
