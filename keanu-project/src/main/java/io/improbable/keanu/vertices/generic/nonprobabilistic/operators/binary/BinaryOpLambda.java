package io.improbable.keanu.vertices.generic.nonprobabilistic.operators.binary;

import io.improbable.keanu.vertices.Vertex;

import java.util.function.BiFunction;

public class BinaryOpLambda<A, B, OUT> extends BinaryOpVertex<A, B, OUT> {

    private BiFunction<A, B, OUT> op;

    public BinaryOpLambda(Vertex<A> a, Vertex<B> b, BiFunction<A, B, OUT> op) {
        super(a, b);
        this.op = op;
    }

    @Override
    protected OUT op(A a, B b) {
        return op.apply(a, b);
    }

}
