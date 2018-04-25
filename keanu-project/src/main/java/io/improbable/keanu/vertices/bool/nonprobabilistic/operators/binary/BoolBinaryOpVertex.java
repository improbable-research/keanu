package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.NonProbabilisticBool;

public abstract class BoolBinaryOpVertex<A, B> extends NonProbabilisticBool {

    protected final Vertex<A> a;
    protected final Vertex<B> b;

    public BoolBinaryOpVertex(Vertex<A> a, Vertex<B> b) {
        this.a = a;
        this.b = b;
        setParents(a, b);
    }

    @Override
    public Boolean sample() {
        return op(a.sample(), b.sample());
    }

    @Override
    public Boolean getDerivedValue() {
        return op(a.getValue(), b.getValue());
    }

    protected abstract Boolean op(A a, B b);
}