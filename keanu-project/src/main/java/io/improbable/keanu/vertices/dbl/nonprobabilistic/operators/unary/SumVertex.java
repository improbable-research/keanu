package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import java.util.Map;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.probabilistic.Differentiable;

public class SumVertex extends DoubleUnaryOpVertex implements Differentiable {

    /**
     * Performs a sum across each value stored in a vertex
     *
     * @param inputVertex the vertex to have its values summed
     */
    public SumVertex(DoubleVertex inputVertex) {
        super(Tensor.SCALAR_SHAPE, inputVertex);
    }

    @Override
    protected DoubleTensor op(DoubleTensor a) {
        return DoubleTensor.scalar(a.sum());
    }

    @Override
    public DualNumber calculateDualNumber(Map<IVertex, DualNumber> dualNumbers) {
        return dualNumbers.get(inputVertex).sum();
    }
}
