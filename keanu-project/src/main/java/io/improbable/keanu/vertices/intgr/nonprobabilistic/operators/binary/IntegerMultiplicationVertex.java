package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class IntegerMultiplicationVertex extends IntegerBinaryOpVertex {

    /**
     * Multiplies one vertex by another
     *
     * @param a a vertex to be multiplied
     * @param b a vertex to be multiplied
     */
    public IntegerMultiplicationVertex(IntegerVertex a, IntegerVertex b) {
        super(checkHasSingleNonScalarShapeOrAllScalar(a.getShape(), b.getShape()), a, b, IntegerTensor::times);
    }
}
