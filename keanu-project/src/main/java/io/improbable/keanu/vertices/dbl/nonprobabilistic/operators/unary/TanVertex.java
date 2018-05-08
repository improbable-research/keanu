package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

import java.util.Map;

public class TanVertex extends DoubleUnaryOpVertex {

    public TanVertex(DoubleVertex inputVertex) {
        super(inputVertex);
    }

    public TanVertex(double inputValue) {
        this(new ConstantDoubleVertex(inputValue));
    }

    @Override
    protected Double op(Double a) {
        return Math.tan(a);
    }

    @Override
    public DualNumber calculateDualNumber(Map<Vertex, DualNumber> dualNumbers) {
        DualNumber inputDualNumber = dualNumbers.get(inputVertex);
        double dTan = 1 / Math.pow(Math.cos(inputVertex.getValue()), 2);
        PartialDerivatives outputPartialDerivatives = inputDualNumber.getPartialDerivatives().multiplyBy(dTan);
        return new DualNumber(op(inputVertex.getValue()), outputPartialDerivatives);
    }
}
