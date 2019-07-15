package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.tensor.TensorVertex;

import java.util.function.Function;

public class IntegerUnaryOpLambda<T, TENSOR extends Tensor<T, TENSOR>, VERTEX extends TensorVertex<T, TENSOR, VERTEX>>
    extends IntegerUnaryOpVertex<T, TENSOR, VERTEX> implements NonSaveableVertex {

    protected final Function<TENSOR, IntegerTensor> op;

    public IntegerUnaryOpLambda(long[] shape, VERTEX inputVertex, Function<TENSOR, IntegerTensor> op) {
        super(shape, inputVertex);
        this.op = op;
    }

    public IntegerUnaryOpLambda(VERTEX inputVertex, Function<TENSOR, IntegerTensor> op) {
        this(inputVertex.getShape(), inputVertex, op);
    }

    @Override
    protected IntegerTensor op(TENSOR value) {
        return op.apply(value);
    }
}
