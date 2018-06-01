package io.improbable.keanu.vertices.booltensor.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.booltensor.nonprobabilistic.NonProbabilisticBool;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

public abstract class BoolUnaryOpVertex<A extends Tensor> extends NonProbabilisticBool {

    protected final Vertex<A> a;

    public BoolUnaryOpVertex(Vertex<A> a) {
        this.a = a;
        setParents(a);
    }

    @Override
    public BooleanTensor sample(KeanuRandom random) {
        return op(a.sample(random));
    }

    @Override
    public BooleanTensor getDerivedValue() {
        return op(a.getValue());
    }

    protected abstract BooleanTensor op(A a);
}