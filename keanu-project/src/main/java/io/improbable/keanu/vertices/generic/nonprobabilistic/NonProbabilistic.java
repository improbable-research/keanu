package io.improbable.keanu.vertices.generic.nonprobabilistic;

import java.util.function.Function;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.update.NonProbabilisticValueUpdater;

public abstract class NonProbabilistic<T> extends Vertex<T> {

    private Function<Vertex<T>, T> updateFunction;

    public NonProbabilistic(Function<Vertex<T>, T> updateFunction) {
        super(new NonProbabilisticValueUpdater<>(updateFunction));
        this.updateFunction = updateFunction;
    }

    @Override
    public boolean matchesObservation() {
        return updateFunction.apply(this).equals(getValue());
    }
}
