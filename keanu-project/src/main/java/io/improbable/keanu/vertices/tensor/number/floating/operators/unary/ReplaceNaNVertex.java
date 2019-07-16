package io.improbable.keanu.vertices.tensor.number.floating.operators.unary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.FloatingPointTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.tensor.TensorVertex;
import io.improbable.keanu.vertices.tensor.UnaryTensorOpVertex;
import io.improbable.keanu.vertices.tensor.number.NumberTensorVertex;

public class ReplaceNaNVertex<T extends Number, TENSOR extends FloatingPointTensor<T, TENSOR>, VERTEX extends NumberTensorVertex<T, TENSOR, VERTEX>>
    extends UnaryTensorOpVertex<T, TENSOR, VERTEX> implements NonProbabilisticVertex<TENSOR, VERTEX> {

    private static final String REPLACE_WITH_VALUE = "REPLACE_WITH_VALUE";
    private final T replaceWithValue;

    @ExportVertexToPythonBindings
    public ReplaceNaNVertex(@LoadVertexParam(INPUT_NAME) TensorVertex<T, TENSOR, VERTEX> inputVertex,
                            @LoadVertexParam(REPLACE_WITH_VALUE) T replaceWithValue) {
        super(inputVertex);
        this.replaceWithValue = replaceWithValue;
    }

    @Override
    protected TENSOR op(TENSOR tensor) {
        return tensor.replaceNaN(replaceWithValue);
    }

    @SaveVertexParam(REPLACE_WITH_VALUE)
    public T getReplaceWithValue() {
        return replaceWithValue;
    }
}