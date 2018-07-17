package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import static io.improbable.keanu.tensor.TensorShapeValidation.checkHasSingleNonScalarShapeOrAllScalar;

import io.improbable.keanu.vertices.dbl.DoubleVertex;

public class PowerVertex extends DoubleBinaryOpVertex {

    /**
     * Raises a vertex to the power of another
     *
     * @param left the base vertex
     * @param right the exponent vertex
     */
    public PowerVertex(DoubleVertex left, DoubleVertex right) {
        super(checkHasSingleNonScalarShapeOrAllScalar(left.getShape(), right.getShape()), left, right,
            (l, r) -> l.pow(r),
            (l, r) -> l.pow(r));
    }

    public DoubleVertex getBase(){
        return super.getLeft();
    }

    public DoubleVertex getExponent(){
        return super.getRight();
    }
}
