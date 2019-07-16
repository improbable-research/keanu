package io.improbable.keanu.vertices.tensor.number.fixed.operators.unary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.FixedPointTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.tensor.BinaryTensorOpVertex;
import io.improbable.keanu.vertices.tensor.TensorVertex;
import io.improbable.keanu.vertices.tensor.number.NumberTensorVertex;

public class ModVertex<T extends Number, TENSOR extends FixedPointTensor<T, TENSOR>, VERTEX extends NumberTensorVertex<T, TENSOR, VERTEX>>
    extends BinaryTensorOpVertex<T, TENSOR, VERTEX> implements NonProbabilisticVertex<TENSOR, VERTEX> {

    @ExportVertexToPythonBindings
    public ModVertex(@LoadVertexParam(LEFT_NAME) TensorVertex<T, TENSOR, VERTEX> left,
                     @LoadVertexParam(RIGHT_NAME) TensorVertex<T, TENSOR, VERTEX> right) {
        super(left, right, left.ofType());
    }

    @Override
    protected TENSOR op(TENSOR l, TENSOR r) {
        return l.mod(r);
    }
}
