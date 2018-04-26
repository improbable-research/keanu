package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

import java.util.Map;

public class DifferenceVertex extends DoubleBinaryOpVertex {

    public DifferenceVertex(DoubleVertex a, DoubleVertex b) {
        super(a, b);
    }

    @Override
    public DualNumber calcDualNumber(Map<Vertex, DualNumber> dualNumberMap) {
        final DualNumber aDual = dualNumberMap.get(a);
        final DualNumber bDual = dualNumberMap.get(b);
        return aDual.subtract(bDual);
    }

    protected Double op(Double a, Double b) {
        return a - b;
    }
}
