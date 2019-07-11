package io.improbable.keanu.vertices.number.operators.binary;

import io.improbable.keanu.annotation.DisplayInformationForOutput;
import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.number.NumberTensorVertex;
import io.improbable.keanu.vertices.tensor.BinaryTensorOpVertex;
import io.improbable.keanu.vertices.tensor.TensorVertex;

@DisplayInformationForOutput(displayName = "-")
public class NumberDifferenceVertex<T extends Number, TENSOR extends NumberTensor<T, TENSOR>, VERTEX extends NumberTensorVertex<T, TENSOR, VERTEX>>
    extends BinaryTensorOpVertex<T, TENSOR, VERTEX> implements NonProbabilisticVertex<TENSOR, VERTEX> {

    @ExportVertexToPythonBindings
    public NumberDifferenceVertex(@LoadVertexParam(LEFT_NAME) TensorVertex<T, TENSOR, VERTEX> left,
                                  @LoadVertexParam(RIGHT_NAME) TensorVertex<T, TENSOR, VERTEX> right) {
        super(left, right);
    }

    @Override
    protected TENSOR op(TENSOR l, TENSOR r) {
        return l.minus(r);
    }
}
