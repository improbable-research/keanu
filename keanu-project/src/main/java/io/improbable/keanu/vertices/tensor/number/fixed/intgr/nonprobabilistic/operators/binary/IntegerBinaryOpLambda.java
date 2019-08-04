package io.improbable.keanu.vertices.tensor.number.fixed.intgr.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.tensor.TensorVertex;

import java.util.function.BiFunction;

import static io.improbable.keanu.tensor.TensorShape.getBroadcastResultShape;


public class IntegerBinaryOpLambda<
    A, TENSORA extends Tensor<A, TENSORA>, VERTEXA extends TensorVertex<A, TENSORA, VERTEXA>,
    B, TENSORB extends Tensor<B, TENSORB>, VERTEXB extends TensorVertex<B, TENSORB, VERTEXB>
    > extends IntegerBinaryOpVertex<A, TENSORA, VERTEXA, B, TENSORB, VERTEXB> implements NonSaveableVertex {

    protected final BiFunction<TENSORA, TENSORB, IntegerTensor> op;

    public IntegerBinaryOpLambda(long[] shape,
                                 TensorVertex<A, TENSORA, VERTEXA> left,
                                 TensorVertex<B, TENSORB, VERTEXB> right,
                                 BiFunction<TENSORA, TENSORB, IntegerTensor> op) {
        super(shape, left, right);
        this.op = op;
    }

    public IntegerBinaryOpLambda(TensorVertex<A, TENSORA, VERTEXA> left,
                                 TensorVertex<B, TENSORB, VERTEXB> right,
                                 BiFunction<TENSORA, TENSORB, IntegerTensor> op) {
        this(getBroadcastResultShape(left.getShape(), right.getShape()), left, right, op);
    }

    @Override
    protected IntegerTensor op(TENSORA l, TENSORB r) {
        return op.apply(l, r);
    }
}
