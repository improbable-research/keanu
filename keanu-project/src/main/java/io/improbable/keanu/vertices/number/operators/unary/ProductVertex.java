package io.improbable.keanu.vertices.number.operators.unary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.number.NumberTensorVertex;
import io.improbable.keanu.vertices.tensor.TensorVertex;
import io.improbable.keanu.vertices.tensor.UnaryTensorOpVertex;

public class ProductVertex<T extends Number, TENSOR extends NumberTensor<T, TENSOR>, VERTEX extends NumberTensorVertex<T, TENSOR, VERTEX>>
    extends UnaryTensorOpVertex<T, TENSOR, VERTEX> implements NonProbabilisticVertex<TENSOR, VERTEX> {

    private final static String OVER_DIMENSIONS = "overDimensions";
    private final int[] overDimensions;

    @ExportVertexToPythonBindings
    public ProductVertex(@LoadVertexParam(INPUT_NAME) TensorVertex<T, TENSOR, VERTEX> inputVertex,
                         @LoadVertexParam(OVER_DIMENSIONS) int[] overDimensions) {
        super(inputVertex, inputVertex.ofType());
        this.overDimensions = overDimensions;
    }

    public ProductVertex(TensorVertex<T, TENSOR, VERTEX> inputVertex) {
        super(new long[0], inputVertex, inputVertex.ofType());
        this.overDimensions = null;
    }

    @Override
    protected TENSOR op(TENSOR value) {
        if (overDimensions != null) {
            return value.product();
        } else {
            return value.product(overDimensions);
        }
    }

    @SaveVertexParam(OVER_DIMENSIONS)
    public int[] getOverDimensions() {
        return overDimensions;
    }
}
