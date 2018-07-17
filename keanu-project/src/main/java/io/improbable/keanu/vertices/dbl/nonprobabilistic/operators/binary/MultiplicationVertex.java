package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;

import io.improbable.keanu.vertices.dbl.DoubleVertex;

public class MultiplicationVertex extends DoubleBinaryOpVertex {

    /**
     * Multiplies one vertex by another
     *
     * @param left vertex to be multiplied
     * @param right vertex to be multiplied
     */
    public MultiplicationVertex(DoubleVertex left, DoubleVertex right) {
        super(checkHasSingleNonScalarShapeOrAllScalar(left.getShape(), right.getShape()), left, right,
            (l, r) -> l.times(r),
            (l, r) -> l.multiplyBy(r));
    }
}
