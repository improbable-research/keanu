package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Vertex;

import java.util.function.BiFunction;

public abstract class BooleanBinaryOpLambda<A extends Tensor, B extends Tensor> extends BooleanBinaryOpVertex<A, B> {

    private final BiFunction<A, B, BooleanTensor> boolOp;

    public BooleanBinaryOpLambda(long[] shape, Vertex<A> a, Vertex<B> b, BiFunction<A, B, BooleanTensor> boolOp) {
        super(shape, a, b);
        this.boolOp = boolOp;
    }

    protected BooleanTensor op(A a, B b) {
        return boolOp.apply(a, b);
    }
}
