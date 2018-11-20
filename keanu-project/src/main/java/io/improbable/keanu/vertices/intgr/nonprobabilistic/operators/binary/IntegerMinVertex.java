package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.LoadParentVertex;
import io.improbable.keanu.vertices.SaveableVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class IntegerMinVertex extends IntegerBinaryOpVertex implements SaveableVertex {

    /**
     * Finds the minimum between two vertices
     *
     * @param left  one of the vertices to find the minimum of
     * @param right one of the vertices to find the minimum of
     */
    public IntegerMinVertex(@LoadParentVertex(LEFT_NAME) IntegerVertex left, @LoadParentVertex(RIGHT_NAME) IntegerVertex right) {
        super(left, right);
    }

    @Override
    protected IntegerTensor op(IntegerTensor l, IntegerTensor r) {
        return IntegerTensor.min(l, r);
    }

}
