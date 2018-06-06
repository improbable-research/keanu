package io.improbable.keanu.vertices.generic.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.Vertex;

import java.util.function.BiFunction;

public class BinaryOpLambda<A extends Tensor, B extends Tensor, C extends Tensor> extends BinaryOpVertex<A, B, C> {

    private BiFunction<A, B, C> op;

    public BinaryOpLambda(Vertex<A> a,
                          Vertex<B> b,
                          BiFunction<A, B, C> op) {
        super(a, b);
        this.op = op;
    }

    @Override
    protected C op(A a, B b) {
        return op.apply(a, b);
    }

}
