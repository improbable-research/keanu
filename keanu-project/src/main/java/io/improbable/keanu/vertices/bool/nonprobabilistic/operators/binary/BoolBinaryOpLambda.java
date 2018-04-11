package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary;

import io.improbable.keanu.vertices.Vertex;

import java.util.function.BiFunction;

public abstract class BoolBinaryOpLambda<A, B> extends BoolBinaryOpVertex<A, B> {

    private final BiFunction<A, B, Boolean> boolOp;

    public BoolBinaryOpLambda(Vertex<A> a, Vertex<B> b, BiFunction<A, B, Boolean> boolOp) {
        super(a, b);
        this.boolOp = boolOp;
    }

    protected Boolean op(A a, B b) {
        return boolOp.apply(a, b);
    }
}

