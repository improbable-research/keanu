package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class IntegerTakeVertex extends IntegerUnaryOpVertex {
    /**
     * A vertex that extracts a scalar at a given index
     *
     * @param inputVertex the input vertex to extract from
     * @param index the index to extract at
     */
    public IntegerTakeVertex(IntegerVertex inputVertex, int... index) {
        super(Tensor.SCALAR_SHAPE, inputVertex,
            a -> IntegerTensor.scalar(a.getValue(index)));
        TensorShapeValidation.checkIndexIsValid(inputVertex.getShape(), index);
    }
}
