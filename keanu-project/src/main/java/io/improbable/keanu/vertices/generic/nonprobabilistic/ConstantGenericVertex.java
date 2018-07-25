package io.improbable.keanu.vertices.generic.nonprobabilistic;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.Observable;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.BooleanBinaryOpVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.generic.GenericVertex;
import io.improbable.keanu.vertices.update.NonProbabilisticValueUpdater;

public class ConstantGenericVertex<T extends Tensor> extends GenericVertex<T> {

    public ConstantGenericVertex(T value) {
        super(
            new NonProbabilisticValueUpdater<>(v -> v.getValue()),
            Observable.observableTypeFor(ConstantGenericVertex.class)
        );
        setValue(value);
    }

    @Override
    public T sample(KeanuRandom random) {
        return getValue();
    }

    public BooleanVertex equalTo(Vertex<T> rhs) {
        return new BooleanBinaryOpVertex<>(this, rhs, (a, b) -> a.elementwiseEquals(b));
    }

    public BooleanVertex notEqualTo(Vertex<T> rhs) {
        return new BooleanBinaryOpVertex<>(this, rhs, (a, b) -> a.elementwiseEquals(b).not());
    }
}
