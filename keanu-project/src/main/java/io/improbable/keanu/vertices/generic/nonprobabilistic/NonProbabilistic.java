package io.improbable.keanu.vertices.generic.nonprobabilistic;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.update.NonProbabilisticValueUpdater;
import io.improbable.keanu.vertices.update.ValueUpdater;

import java.util.Map;
import java.util.function.Function;
import java.util.function.Supplier;

public abstract class NonProbabilistic<T> extends Vertex<T> {

    private Function<Vertex<T>, T> updateFunction;

    public NonProbabilistic(Function<Vertex<T>, T> updateFunction) {
        super(new NonProbabilisticValueUpdater<>(updateFunction));
        this.updateFunction = updateFunction;
    }

    @Override
    public double logProb(T value) {
        return updateFunction.apply(this).equals(value) ? 0.0 : Double.NEGATIVE_INFINITY;
    }

    @Override
    public Map<Long, DoubleTensor> dLogProb(T value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean isProbabilistic() {
        return false;
    }

}
