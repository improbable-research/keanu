package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

public class PowerVertex extends DoubleBinaryOpVertex {

    /**
     * Raises a vertex to the power of another
     *
     * @param left the base vertex
     * @param right the exponent vertex
     */
    public PowerVertex(DoubleVertex left, DoubleVertex right) {
        super(left, right,
            DoubleTensor::pow,
            DualNumber::pow);
    }

    public DoubleVertex getBase(){
        return super.getLeft();
    }

    public DoubleVertex getExponent(){
        return super.getRight();
    }
}
