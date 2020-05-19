package io.improbable.keanu.vertices.tensor.generic.nonprobabilistic.operators.binary;

import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.tensor.generic.GenericVertex;

public abstract class BinaryOpVertex<A, B, C> extends VertexImpl<C, GenericVertex<C>> implements GenericVertex<C>, NonProbabilistic<C> {

    protected final Vertex<A, ?> a;
    protected final Vertex<B, ?> b;

    public BinaryOpVertex(Vertex<A, ?> a, Vertex<B, ?> b) {
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
