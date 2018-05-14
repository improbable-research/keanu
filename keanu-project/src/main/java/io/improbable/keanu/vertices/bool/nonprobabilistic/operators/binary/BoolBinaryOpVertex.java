package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.NonProbabilisticBool;

import java.util.Random;

public abstract class BoolBinaryOpVertex<A, B> extends NonProbabilisticBool {

    protected final Vertex<A> a;
    protected final Vertex<B> b;

    public BoolBinaryOpVertex(Vertex<A> a, Vertex<B> b) {
        this.a = a;
        this.b = b;
        setParents(a, b);
    }

    @Override
    public Boolean sample(Random random) {
        return op(a.sample(random), b.sample(random));
    }

    @Override
    public Boolean getDerivedValue() {
        return op(a.getValue(), b.getValue());
    }

    protected abstract Boolean op(A a, B b);
}