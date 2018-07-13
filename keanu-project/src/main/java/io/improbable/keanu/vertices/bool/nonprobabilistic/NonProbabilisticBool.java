package io.improbable.keanu.vertices.bool.nonprobabilistic;

import java.util.function.Function;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.update.NonProbabilisticValueUpdater;

public abstract class NonProbabilisticBool extends BoolVertex {

    private Function<Vertex<BooleanTensor>, BooleanTensor> updateFunction;

    public NonProbabilisticBool(Function<Vertex<BooleanTensor>, BooleanTensor> updateFunction) {
        super(new NonProbabilisticValueUpdater<>(updateFunction));
        this.updateFunction = updateFunction;
    }

    @Override
    public boolean matchesObservation() {
        return updateFunction.apply(this).elementwiseEquals(getValue()).allTrue();
    }
}
