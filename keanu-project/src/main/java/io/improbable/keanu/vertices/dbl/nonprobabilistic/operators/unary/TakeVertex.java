package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

public class TakeVertex extends DoubleUnaryOpVertex {

    /**
     * A vertex that extracts a scalar at a given index
     *
     * @param inputVertex the input vertex to extract from
     * @param index the index to extract at
     */
    public TakeVertex(DoubleVertex inputVertex, int... index) {
        super(Tensor.SCALAR_SHAPE, inputVertex,
            a -> DoubleTensor.scalar(a.getValue(index)),
            a -> a.take(index));
        TensorShapeValidation.checkIndexIsValid(inputVertex.getShape(), index);
    }
}
