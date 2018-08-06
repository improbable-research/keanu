package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;

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
        super(checkHasSingleNonScalarShapeOrAllScalar(a.getShape(), b.getShape()), a, b, IntegerTensor::minus);
    }
}
