package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;

public class IntegerAdditionVertex extends IntegerBinaryOpVertex {

    /**
     * Adds one vertex to another
     *
     * @param a a vertex to add
     * @param b a vertex to add
     */
    public IntegerAdditionVertex(IntegerVertex a, IntegerVertex b) {
        super(checkHasSingleNonScalarShapeOrAllScalar(a.getShape(), b.getShape()), a, b);
    }

    @Override
    protected IntegerTensor op(IntegerTensor a, IntegerTensor b) {
        return a.plus(b);
    }
}
