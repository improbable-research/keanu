package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.LoadParentVertex;
import io.improbable.keanu.vertices.SaveableVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class IntegerPowerVertex extends IntegerBinaryOpVertex implements SaveableVertex {
    /**
     * Raises one vertex to the power of another
     *
     * @param a the base vertex
     * @param b the exponent vertex
     */
    public IntegerPowerVertex(@LoadParentVertex(A_NAME) IntegerVertex a, @LoadParentVertex(B_NAME) IntegerVertex b) {
        super(a, b);
    }

    @Override
    protected IntegerTensor op(IntegerTensor l, IntegerTensor r) {
        return l.pow(r);
    }
}
