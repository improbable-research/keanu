package io.improbable.keanu.vertices.intgrtensor.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.intgrtensor.IntegerVertex;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;

public class IntegerDifferenceVertex extends IntegerBinaryOpVertex {

    public IntegerDifferenceVertex(IntegerVertex a, IntegerVertex b) {
        super(checkHasSingleNonScalarShapeOrAllScalar(a.getShape(), b.getShape()), a, b);
    }

    protected IntegerTensor op(IntegerTensor a, IntegerTensor b) {
        return a.minus(b);
    }
}
