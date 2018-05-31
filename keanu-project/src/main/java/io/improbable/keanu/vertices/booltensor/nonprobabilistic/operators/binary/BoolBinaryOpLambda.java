package io.improbable.keanu.vertices.booltensor.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Vertex;

import java.util.function.BiFunction;

public abstract class BoolBinaryOpLambda<A extends Tensor, B extends Tensor> extends BoolBinaryOpVertex<A, B> {

    private final BiFunction<A, B, BooleanTensor> boolOp;

    public BoolBinaryOpLambda(Vertex<A> a, Vertex<B> b, BiFunction<A, B, BooleanTensor> boolOp) {
        super(a, b);
        this.boolOp = boolOp;
    }

    protected BooleanTensor op(A a, B b) {
        return boolOp.apply(a, b);
    }
}

