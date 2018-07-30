package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

import java.util.Map;

public class PluckVertex extends DoubleUnaryOpVertex {

    private final int[] index;

    /**
     * A vertex that extracts a scalar at a given index
     *
     * @param inputVertex the input vertex to extract from
     * @param index the index to extract at
     */
    public PluckVertex(DoubleVertex inputVertex, int... index) {
        super(Tensor.SCALAR_SHAPE, inputVertex);
        TensorShapeValidation.checkIndexIsValid(inputVertex.getShape(), index);
        this.index = index;
    }

    @Override
    protected DoubleTensor op(DoubleTensor a) {
        return DoubleTensor.scalar(a.getValue(index));
    }

    @Override
    protected DualNumber calculateDualNumber(Map<Vertex, DualNumber> dualNumbers) {
        return dualNumbers.get(inputVertex).pluck(index);
    }
}
