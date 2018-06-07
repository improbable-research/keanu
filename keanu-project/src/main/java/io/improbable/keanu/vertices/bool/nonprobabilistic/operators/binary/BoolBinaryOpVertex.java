package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.NonProbabilisticBool;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public abstract class BoolBinaryOpVertex<A extends Tensor, B extends Tensor> extends NonProbabilisticBool {

    protected final Vertex<A> a;
    protected final Vertex<B> b;

    public BoolBinaryOpVertex(int[] shape, Vertex<A> a, Vertex<B> b) {
        this.a = a;
        this.b = b;
        setParents(a, b);
        setValue(BooleanTensor.placeHolder(shape));
    }

    @Override
    public BooleanTensor sample(KeanuRandom random) {
        return op(a.sample(random), b.sample(random));
    }

    @Override
    public BooleanTensor getDerivedValue() {
        return op(a.getValue(), b.getValue());
    }

    protected abstract BooleanTensor op(A a, B b);
}