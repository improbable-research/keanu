package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

import java.util.Map;


public class MultiplicationVertex extends DoubleBinaryOpVertex {

    public MultiplicationVertex(DoubleVertex a, DoubleVertex b) {
        super(a, b);
    }

    @Override
    public DualNumber calculateDualNumber(Map<Vertex, DualNumber> dualNumberMap) {
        DualNumber aDual = dualNumberMap.get(a);
        DualNumber bDual = dualNumberMap.get(b);
        return aDual.multiplyBy(bDual);
    }

    @Override
    protected Double op(Double a, Double b) {
        return a * b;
    }
}
