package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.NonProbabilisticBool;

public abstract class BoolUnaryOpVertex<A> extends NonProbabilisticBool {

    protected final Vertex<A> a;

    public BoolUnaryOpVertex(Vertex<A> a) {
        this.a = a;
        setParents(a);
    }

    @Override
    public Boolean sample() {
        return op(a.sample());
    }

    @Override
    public Boolean getDerivedValue() {
        return op(a.getValue());
    }

    protected abstract Boolean op(A a);
}