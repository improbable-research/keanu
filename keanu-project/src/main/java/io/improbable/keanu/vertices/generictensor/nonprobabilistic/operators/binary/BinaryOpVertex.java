package io.improbable.keanu.vertices.generictensor.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.generictensor.nonprobabilistic.NonProbabilistic;

public abstract class BinaryOpVertex<A, B, C,
    TA extends Tensor<A>, TB extends Tensor<B>, TC extends Tensor<C>> extends NonProbabilistic<C, TC> {

    protected final Vertex<TA> a;
    protected final Vertex<TB> b;

    public BinaryOpVertex(Vertex<TA> a, Vertex<TB> b) {
        this.a = a;
        this.b = b;
        setParents(a, b);
    }

    @Override
    public TC sample(KeanuRandom random) {
        return op(a.sample(random), b.sample(random));
    }

    public TC getDerivedValue() {
        return op(a.getValue(), b.getValue());
    }

    protected abstract TC op(TA a, TB b);
}

