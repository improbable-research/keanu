package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.Infinitesimal;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.DoubleUnaryOpVertex;

public class ArcTan2Vertex extends DoubleBinaryOpVertex {

    public ArcTan2Vertex(DoubleVertex a, DoubleVertex b) {
        super(a, b);
    }

    public ArcTan2Vertex(double a, double b) {
        this(new ConstantDoubleVertex(a), new ConstantDoubleVertex(b));
    }

    @Override
    protected Double op(Double a, Double b) {
        return Math.atan2(a, b);
    }

    @Override
    public DualNumber getDualNumber() {
        DualNumber aDual = a.getDualNumber();
        DualNumber bDual = b.getDualNumber();

        Infinitesimal thisInfA = aDual.getInfinitesimal().multiplyBy(b.getValue() / (Math.pow(b.getValue(), 2) * Math.pow(a.getValue(), 2)));
        Infinitesimal thisInfB = bDual.getInfinitesimal().multiplyBy(- (a.getValue() / (Math.pow(b.getValue(), 2) * Math.pow(a.getValue(), 2))));
        Infinitesimal newInf = thisInfA.add(thisInfB);
        return new DualNumber(op(a.getValue(), b.getValue()), newInf);
    }

}
