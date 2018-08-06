package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.intgr.IntegerVertex;


public class IntegerDivisionVertex extends IntegerBinaryOpVertex {

    /**
     * Divides one vertex by another
     *
     * @param a a vertex to be divided
     * @param b a vertex to divide by
     */
    public IntegerDivisionVertex(IntegerVertex a, IntegerVertex b) {
        super(checkHasSingleNonScalarShapeOrAllScalar(a.getShape(), b.getShape()), a, b, IntegerTensor::div);
    }
}
