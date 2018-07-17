package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary;

import java.util.function.Function;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Observable;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.update.NonProbabilisticValueUpdater;

public class BoolUnaryOpVertex<T extends Tensor> extends BoolVertex {

    protected final Vertex<T> a;
    private final Function<T, BooleanTensor> op;

    public BoolUnaryOpVertex(Vertex<T> a, Function<T, BooleanTensor> op) {
        this(a.getShape(), a, op);
    }

    public BoolUnaryOpVertex(int[] shape, Vertex<T> a, Function<T, BooleanTensor> op) {
        super(
            new NonProbabilisticValueUpdater<>(v -> op.apply(a.getValue())),
            Observable.observableTypeFor(BoolUnaryOpVertex.class)
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