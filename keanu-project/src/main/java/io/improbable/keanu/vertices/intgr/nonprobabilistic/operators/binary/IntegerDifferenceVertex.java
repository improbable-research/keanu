package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary;

import io.improbable.keanu.annotation.DisplayInformationForOutput;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class IntegerDifferenceVertex extends IntegerBinaryOpVertex {

    /**
     * Subtracts one vertex from another
     *
     * @param a the vertex to be subtracted from
     * @param b the vertex to subtract
     */
    @DisplayInformationForOutput(displayName = "-")
    public IntegerDifferenceVertex(IntegerVertex a, IntegerVertex b) {
        super(a, b);
    }

    @Override
    protected IntegerTensor op(IntegerTensor l, IntegerTensor r) {
        return l.minus(r);
    }
}
