package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;

public class AdditionVertex extends DoubleBinaryOpVertex {

    /**
     * Adds one vertex to another
     *
     * @param left a vertex to add
     * @param right a vertex to add
     */
    public AdditionVertex(DoubleVertex left, DoubleVertex right) {
        super(left, right,
            (l,r) -> l.plus(r),
            (l,r) -> l.plus(r)
        );
    }
}
