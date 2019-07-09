package io.improbable.keanu.vertices.generic.nonprobabilistic.operators.binary;

import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.NonSaveableVertex;

import java.util.function.BiFunction;

public class BinaryOpLambda<A, B, C> extends BinaryOpVertex<A, B, C> implements NonSaveableVertex {

    private BiFunction<A, B, C> op;

    public BinaryOpLambda(IVertex<A> a,
                          IVertex<B> b,
                          BiFunction<A, B, C> op) {
        super(a, b);
        this.op = op;
    }

    @Override
    protected C op(A a, B b) {
        return op.apply(a, b);
    }

}
