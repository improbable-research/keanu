package io.improbable.keanu.vertices.generic.nonprobabilistic.operators.binary;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.NonProbabilistic;

public abstract class BinaryOpVertex<A, B, C> extends NonProbabilistic<C> {

    protected final Vertex<A> a;
    protected final Vertex<B> b;

    public BinaryOpVertex(Vertex<A> a, Vertex<B> b) {
        this.a = a;
        this.b = b;
        setParents(a, b);
    }

    @Override
    public C sample() {
        return op(a.sample(), b.sample());
    }

    public C getDerivedValue() {
        return op(a.getValue(), b.getValue());
    }

    protected abstract C op(A a, B b);
}

