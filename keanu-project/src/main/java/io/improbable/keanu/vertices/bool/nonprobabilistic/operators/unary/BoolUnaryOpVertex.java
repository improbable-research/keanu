package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.NonProbabilisticBool;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public abstract class BoolUnaryOpVertex<A extends Tensor> extends NonProbabilisticBool {

    protected final Vertex<A> a;

    public BoolUnaryOpVertex(int[] shape, Vertex<A> a) {
        this.a = a;
        setParents(a);
        setValue(BooleanTensor.placeHolder(shape));
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