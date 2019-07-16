package io.improbable.keanu.vertices.tensor.number.operators.unary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.tensor.TensorVertex;
import io.improbable.keanu.vertices.tensor.UnaryTensorOpVertex;
import io.improbable.keanu.vertices.tensor.number.NumberTensorVertex;


public class AbsVertex<T extends Number, TENSOR extends NumberTensor<T, TENSOR>, VERTEX extends NumberTensorVertex<T, TENSOR, VERTEX>>
    extends UnaryTensorOpVertex<T, TENSOR, VERTEX> implements NonProbabilisticVertex<TENSOR, VERTEX> {

    /**
     * Takes the absolute of a vertex
     *
     * @param inputVertex the vertex
     */
    @ExportVertexToPythonBindings
    public AbsVertex(@LoadVertexParam(INPUT_NAME) TensorVertex<T, TENSOR, VERTEX> inputVertex) {
        super(inputVertex, inputVertex.ofType());
    }

    @Override
    protected TENSOR op(TENSOR value) {
        return value.abs();
    }

}
