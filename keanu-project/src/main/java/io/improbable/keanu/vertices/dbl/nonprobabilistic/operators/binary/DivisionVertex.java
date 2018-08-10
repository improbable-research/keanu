package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

public class DivisionVertex extends DoubleBinaryOpVertex {
    /**
     * Divides one vertex by another
     *
     * @param left the vertex to be divided
     * @param right the vertex to divide
     */
    public DivisionVertex(DoubleVertex left, DoubleVertex right) {
        super(left, right,
            DoubleTensor::div,
            DualNumber::div);
    }
}
