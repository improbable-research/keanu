package io.improbable.keanu.vertices.number.operators.unary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.number.NumberTensorVertex;
import io.improbable.keanu.vertices.tensor.TensorVertex;
import io.improbable.keanu.vertices.tensor.UnaryTensorOpVertex;

public class CumSumVertex<T extends Number, TENSOR extends NumberTensor<T, TENSOR>, VERTEX extends NumberTensorVertex<T, TENSOR, VERTEX>>
    extends UnaryTensorOpVertex<T, TENSOR, VERTEX> implements NonProbabilisticVertex<TENSOR, VERTEX> {

    private final static String REQUESTED_DIMENSION = "requestedDimension";
    private final int requestedDimension;

    @ExportVertexToPythonBindings
    public CumSumVertex(@LoadVertexParam(INPUT_NAME) TensorVertex<T, TENSOR, VERTEX> inputVertex,
                        @LoadVertexParam(REQUESTED_DIMENSION) int requestedDimension) {
        super(inputVertex, inputVertex.ofType());
        this.requestedDimension = requestedDimension;
    }

    @Override
    protected TENSOR op(TENSOR value) {
        return value.cumSum(requestedDimension);
    }

    @SaveVertexParam(REQUESTED_DIMENSION)
    public int getRequestedDimension() {
        return this.requestedDimension;
    }
}
