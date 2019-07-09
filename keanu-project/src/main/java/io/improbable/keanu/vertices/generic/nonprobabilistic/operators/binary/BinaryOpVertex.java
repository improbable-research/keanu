package io.improbable.keanu.vertices.generic.nonprobabilistic.operators.binary;

import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.generic.GenericVertex;

public abstract class BinaryOpVertex<A, B, C> extends GenericVertex<C> implements NonProbabilistic<C> {

    protected final IVertex<A> a;
    protected final IVertex<B> b;

    public BinaryOpVertex(IVertex<A> a, IVertex<B> b) {
        this.a = a;
        this.b = b;
        setParents(a, b);
    }

    @Override
    public C calculate() {
        return op(a.getValue(), b.getValue());
    }

    protected abstract C op(A a, B b);
}
