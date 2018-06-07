package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

import java.util.Map;

public class LogVertex extends DoubleUnaryOpVertex {

    public LogVertex(DoubleVertex inputVertex) {
        super(inputVertex.getShape(), inputVertex);
    }

    protected DoubleTensor op(DoubleTensor a) {
        return a.log();
    }

    @Override
    protected DualNumber calculateDualNumber(Map<Vertex, DualNumber> dualNumbers) {
        return dualNumbers.get(inputVertex).log();
    }
}
