package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Observable;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.update.NonProbabilisticValueUpdater;

public abstract class BoolUnaryOpVertex<A extends Tensor> extends BoolVertex {

    protected final Vertex<A> a;

    public BoolUnaryOpVertex(int[] shape, Vertex<A> a) {
        super(
            new NonProbabilisticValueUpdater<>(v -> ((BoolUnaryOpVertex)v).op(a.getValue())),
            Observable.observableTypeFor(BoolUnaryOpVertex.class)
        );
        this.a = a;
        setParents(a);
        setValue(BooleanTensor.placeHolder(shape));
    }

    @Override
    public BooleanTensor sample(KeanuRandom random) {
        return op(a.sample(random));
    }

    protected abstract BooleanTensor op(A a);
}