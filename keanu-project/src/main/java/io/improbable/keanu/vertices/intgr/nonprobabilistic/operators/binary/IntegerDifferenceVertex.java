package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class IntegerDifferenceVertex extends IntegerBinaryOpVertex {

    /**
     * Subtracts one vertex from another
     *
     * @param a the vertex to be subtracted from
     * @param b the vertex to subtract
     */
    public IntegerDifferenceVertex(IntegerVertex a, IntegerVertex b) {
        super(a, b, IntegerTensor::minus);
    }
}
