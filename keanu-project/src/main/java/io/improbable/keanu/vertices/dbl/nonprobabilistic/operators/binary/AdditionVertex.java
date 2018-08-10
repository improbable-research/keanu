package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

public class AdditionVertex extends DoubleBinaryOpVertex {

    /**
     * Adds one vertex to another
     *
     * @param left a vertex to add
     * @param right a vertex to add
     */
    public AdditionVertex(DoubleVertex left, DoubleVertex right) {
        super(left, right,
            DoubleTensor::plus,
            DualNumber::plus
        );
    }
}
