package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

import java.util.Map;


public class DivisionVertex extends DoubleBinaryOpVertex {

    public DivisionVertex(DoubleVertex a, DoubleVertex b) {
        super(a, b);
    }

    @Override
    public DualNumber calcDualNumber(Map<Vertex, DualNumber> dualNumberMap) {
        DualNumber aDual = dualNumberMap.get(a);
        DualNumber bDual = dualNumberMap.get(b);
        return aDual.divideBy(bDual);
    }

    protected Double op(Double a, Double b) {
        return a / b;
    }
}
