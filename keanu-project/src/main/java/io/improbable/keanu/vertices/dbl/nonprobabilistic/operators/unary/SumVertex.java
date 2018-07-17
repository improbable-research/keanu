package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

public class SumVertex extends DoubleUnaryOpVertex {

    /**
     * Performs a sum across each value stored in a vertex
     *
     * @param inputVertex the vertex to have its values summed
     */
    public SumVertex(DoubleVertex inputVertex) {
        super(Tensor.SCALAR_SHAPE, inputVertex,
            a -> DoubleTensor.scalar(a.sum()),
            DualNumber::sum);
    }
}
