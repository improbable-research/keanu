package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import java.util.Map;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.probabilistic.Differentiable;

public class ExpVertex extends DoubleUnaryOpVertex implements Differentiable {

    /**
     * Calculates the exponential of an input vertex
     *
     * @param inputVertex the vertex
     */
    public ExpVertex(DoubleVertex inputVertex) {
        super(inputVertex.getShape(), inputVertex);
    }

    @Override
    protected DoubleTensor op(DoubleTensor a) {
        return a.exp();
    }

    @Override
    public DualNumber calculateDualNumber(Map<IVertex, DualNumber> dualNumbers) {
        return dualNumbers.get(inputVertex).exp();
    }
}
