package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.LoadParentVertex;
import io.improbable.keanu.vertices.SaveableVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class IntegerMaxVertex extends IntegerBinaryOpVertex implements SaveableVertex {

    /**
     * Finds the maximum between two vertices
     *
     * @param left  one of the vertices to find the maximum of
     * @param right one of the vertices to find the maximum of
     */
    public IntegerMaxVertex(@LoadParentVertex(A_NAME) IntegerVertex left, @LoadParentVertex(B_NAME) IntegerVertex right) {
        super(left, right);
    }

    @Override
    protected IntegerTensor op(IntegerTensor l, IntegerTensor r) {
        return IntegerTensor.max(l, r);
    }
}
