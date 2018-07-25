package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary;

import java.util.function.Function;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Observable;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.update.NonProbabilisticValueUpdater;

public class BooleanUnaryOpVertex<T extends Tensor> extends BooleanVertex {

    protected final Vertex<T> a;
    private final Function<T, BooleanTensor> op;

    public BooleanUnaryOpVertex(Vertex<T> a, Function<T, BooleanTensor> op) {
        this(a.getShape(), a, op);
    }

    public BooleanUnaryOpVertex(int[] shape, Vertex<T> a, Function<T, BooleanTensor> op) {
        super(
            new NonProbabilisticValueUpdater<>(v -> op.apply(a.getValue())),
            Observable.observableTypeFor(BooleanUnaryOpVertex.class)
        );
        this.a = a;
        this.op = op;
        setParents(a);
        setValue(BooleanTensor.placeHolder(shape));
    }

    @Override
    public BooleanTensor sample(KeanuRandom random) {
        return op.apply(a.sample(random));
    }
}