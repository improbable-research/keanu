package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.update.NonProbabilisticValueUpdater;

public abstract class BoolUnaryOpVertex<T extends Tensor> extends BoolVertex {

    protected final Vertex<T> a;

    public BoolUnaryOpVertex(Vertex<T> a) {
        this(a.getShape(), a);
    }

    public BoolUnaryOpVertex(int[] shape, Vertex<T> a) {
        super(new NonProbabilisticValueUpdater<>(v -> ((BoolUnaryOpVertex) v).op(a.getValue())));
        this.a = a;
        setParents(a);
        setValue(BooleanTensor.placeHolder(shape));
    }

    protected abstract BooleanTensor op(T value);

    @Override
    public BooleanTensor sample(KeanuRandom random) {
        return op(a.sample(random));
    }
}