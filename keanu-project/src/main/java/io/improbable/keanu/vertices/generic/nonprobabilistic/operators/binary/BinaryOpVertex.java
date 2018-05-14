package io.improbable.keanu.vertices.generic.nonprobabilistic.operators.binary;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.NonProbabilistic;

import java.util.Random;

public abstract class BinaryOpVertex<A, B, C> extends NonProbabilistic<C> {

    protected final Vertex<A> a;
    protected final Vertex<B> b;

    public BinaryOpVertex(Vertex<A> a, Vertex<B> b) {
        this.a = a;
        this.b = b;
        setParents(a, b);
    }

    @Override
    public C sample(Random random) {
        return op(a.sample(random), b.sample(random));
    }

    public C getDerivedValue() {
        return op(a.getValue(), b.getValue());
    }

    protected abstract C op(A a, B b);
}

