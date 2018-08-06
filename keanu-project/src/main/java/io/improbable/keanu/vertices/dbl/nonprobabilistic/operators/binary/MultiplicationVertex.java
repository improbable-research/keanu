package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;

public class MultiplicationVertex extends DoubleBinaryOpVertex {

    /**
     * Multiplies one vertex by another
     *
     * @param left vertex to be multiplied
     * @param right vertex to be multiplied
     */
    public MultiplicationVertex(DoubleVertex left, DoubleVertex right) {
        super(left, right,
            (l, r) -> l.times(r),
            (l, r) -> l.multiplyBy(r));
    }
}
