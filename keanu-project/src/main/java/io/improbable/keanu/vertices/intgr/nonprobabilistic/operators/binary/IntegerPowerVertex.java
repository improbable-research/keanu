package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class IntegerPowerVertex extends IntegerBinaryOpVertex {
    /**
     * Multiplies one vertex by another
     *
     * @param a the base vertex
     * @param b the exponent vertex
     */
    public IntegerPowerVertex(IntegerVertex a, IntegerVertex b) {
        super(checkHasSingleNonScalarShapeOrAllScalar(a.getShape(), b.getShape()), a, b, IntegerTensor::pow);
    }
}
