package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;

public class DivisionVertex extends DoubleBinaryOpVertex {
    /**
     * Divides one vertex by another
     *
     * @param left the vertex to be divided
     * @param right the vertex to divide
     */
    public DivisionVertex(DoubleVertex left, DoubleVertex right) {
        super(left, right,
            (l, r) -> l.div(r),
            (l, r) -> l.div(r));
    }
}
