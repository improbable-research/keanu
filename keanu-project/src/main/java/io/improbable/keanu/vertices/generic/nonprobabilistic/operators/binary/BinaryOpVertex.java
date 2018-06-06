package io.improbable.keanu.vertices.generic.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.generic.nonprobabilistic.NonProbabilistic;

public abstract class BinaryOpVertex<A extends Tensor, B extends Tensor, C extends Tensor> extends NonProbabilistic<C> {

    protected final Vertex<A> a;
    protected final Vertex<B> b;

    public BinaryOpVertex(Vertex<A> a, Vertex<B> b) {
        this.a = a;
        this.b = b;
        setParents(a, b);
    }

    @Override
    public C sample(KeanuRandom random) {
        return op(a.sample(random), b.sample(random));
    }

    public C getDerivedValue() {
        return op(a.getValue(), b.getValue());
    }

    protected abstract C op(A a, B b);
}

