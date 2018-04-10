package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.Vertex;

import java.util.function.Function;

public abstract class BoolUnaryOpLambda<A> extends BoolUnaryOpVertex<A> {

    private final Function<A, Boolean> boolOp;

    public BoolUnaryOpLambda(Vertex<A> a, Function<A, Boolean> boolOp) {
        super(a);
        this.boolOp = boolOp;
    }

    protected Boolean op(A a) {
        return boolOp.apply(a);
    }
}
