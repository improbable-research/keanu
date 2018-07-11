package io.improbable.keanu.vertices.dbl.nonprobabilistic;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.NonProbabilisticObservationException;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.update.NonProbabilisticValueUpdater;
import io.improbable.keanu.vertices.update.ValueUpdater;

import java.util.Map;
import java.util.function.Function;

public abstract class NonProbabilisticDouble extends DoubleVertex {

    private Function<Vertex<DoubleTensor>, DoubleTensor> updateFunction;

    public NonProbabilisticDouble(Function<Vertex<DoubleTensor>, DoubleTensor> updateFunction) {
        super(new NonProbabilisticValueUpdater<>(updateFunction));
        this.updateFunction = updateFunction;
    }

    /**
     * Observing non-probabilistic values of this type causes the probability
     * of the graph to flatten to 0 in all places that doesn't exactly match
     * the observation. This is so bad that it is actually prohibited by throwing
     * an exception. This is not the case for all types of non-probabilistic
     * observations.
     *
     * @param value the value to be observed
     */
    @Override
    public void observe(DoubleTensor value) {
        throw new NonProbabilisticObservationException();
    }

    @Override
    public double logPdf(DoubleTensor value) {
        return updateFunction.apply(this).elementwiseEquals(getValue()).allTrue() ? 0.0 : Double.NEGATIVE_INFINITY;
    }

    @Override
    public Map<Long, DoubleTensor> dLogPdf(DoubleTensor value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean isProbabilistic() {
        return false;
    }
}
