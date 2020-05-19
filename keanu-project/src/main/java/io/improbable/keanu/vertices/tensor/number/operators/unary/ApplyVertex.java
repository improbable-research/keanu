package io.improbable.keanu.vertices.tensor.number.operators.unary;

import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.tensor.TensorVertex;
import io.improbable.keanu.vertices.tensor.UnaryTensorOpVertex;
import io.improbable.keanu.vertices.tensor.number.NumberTensorVertex;

import java.util.function.Function;

public class ApplyVertex<T extends Number, TENSOR extends NumberTensor<T, TENSOR>, VERTEX extends NumberTensorVertex<T, TENSOR, VERTEX>>
    extends UnaryTensorOpVertex<T, TENSOR, VERTEX> implements NonProbabilisticVertex<TENSOR, VERTEX>, NonSaveableVertex {

    protected final Function<T, T> op;

    public ApplyVertex(long[] shape, TensorVertex<T, TENSOR, VERTEX> inputVertex, Function<T, T> op) {
        super(shape, inputVertex, inputVertex.ofType());
        this.op = op;
    }

    public ApplyVertex(TensorVertex<T, TENSOR, VERTEX> inputVertex, Function<T, T> op) {
        this(inputVertex.getShape(), inputVertex, op);
    }

    @Override
    protected TENSOR op(TENSOR value) {
        return inputVertex.getValue().apply(op);
    }
}
