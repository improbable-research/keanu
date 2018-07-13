package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import java.util.Map;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.probabilistic.Differentiable;

public class LogVertex extends DoubleUnaryOpVertex implements Differentiable {

    /**
     * Returns the natural logarithm, base e, of a vertex
     *
     * @param inputVertex the vertex
     */
    public LogVertex(DoubleVertex inputVertex) {
        super(inputVertex.getShape(), inputVertex);
    }

    protected DoubleTensor op(DoubleTensor a) {
        return a.log();
    }

    @Override
    public DualNumber calculateDualNumber(Map<IVertex, DualNumber> dualNumbers) {
        return dualNumbers.get(inputVertex).log();
    }
}
