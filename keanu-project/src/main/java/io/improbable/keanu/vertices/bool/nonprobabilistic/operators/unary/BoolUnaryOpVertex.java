package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.NonProbabilisticBool;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

public abstract class BoolUnaryOpVertex<A> extends NonProbabilisticBool {

    protected final Vertex<A> a;

    public BoolUnaryOpVertex(Vertex<A> a) {
        this.a = a;
        setParents(a);
    }

    @Override
    public Boolean sample(KeanuRandom random) {
        return op(a.sample(random));
    }

    @Override
    public Boolean getDerivedValue() {
        return op(a.getValue());
    }

    protected abstract Boolean op(A a);
}