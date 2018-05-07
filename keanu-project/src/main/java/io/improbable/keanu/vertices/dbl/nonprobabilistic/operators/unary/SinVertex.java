package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

import java.util.Map;

public class SinVertex extends DoubleUnaryOpVertex {

    public SinVertex(double inputValue) {
        super(new ConstantDoubleVertex(inputValue));
    }

    public SinVertex(DoubleVertex inputVertex) {
        super(inputVertex);
    }

    @Override
    protected Double op(Double a) {
        return Math.sin(a);
    }

    @Override
    public DualNumber calculateDualNumber(Map<Vertex, DualNumber> dualNumbers) {
        return dualNumbers.get(inputVertex).sin();
    }

}
