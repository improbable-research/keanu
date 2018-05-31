package io.improbable.keanu.vertices.booltensor.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.booltensor.nonprobabilistic.NonProbabilisticBool;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

public abstract class BoolBinaryOpVertex<TA extends Tensor, TB extends Tensor> extends NonProbabilisticBool {

    protected final Vertex<TA> a;
    protected final Vertex<TB> b;

    public BoolBinaryOpVertex(Vertex<TA> a, Vertex<TB> b) {
        this.a = a;
        this.b = b;
        setParents(a, b);
    }

    @Override
    public BooleanTensor sample(KeanuRandom random) {
        return op(a.sample(random), b.sample(random));
    }

    @Override
    public BooleanTensor getDerivedValue() {
        return op(a.getValue(), b.getValue());
    }

    protected abstract BooleanTensor op(TA a, TB b);
}