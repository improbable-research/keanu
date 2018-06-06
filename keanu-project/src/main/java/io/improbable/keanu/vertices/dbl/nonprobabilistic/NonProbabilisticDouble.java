package io.improbable.keanu.vertices.dbl.nonprobabilistic;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.NonProbabilisticObservationException;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

import java.util.Map;

public abstract class NonProbabilisticDouble extends DoubleVertex {

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
        return this.getDerivedValue().elementwiseEquals(getValue()).allTrue() ? 0.0 : Double.NEGATIVE_INFINITY;
    }

    @Override
    public Map<Long, DoubleTensor> dLogPdf(DoubleTensor value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean isProbabilistic() {
        return false;
    }

    @Override
    public DoubleTensor updateValue() {
        if (!this.isObserved()) {
            setValue(getDerivedValue());
        }
        return getValue();
    }

    public abstract DoubleTensor getDerivedValue();
}
