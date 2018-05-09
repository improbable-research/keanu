package io.improbable.keanu.vertices.dbl.nonprobabilistic;

import io.improbable.keanu.vertices.NonProbabilisticObservationException;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbltensor.DoubleTensor;

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
    public void observe(Double value) {
        throw new NonProbabilisticObservationException();
    }

    @Override
    public double logPdf(Double value) {
        return this.getDerivedValue().equals(value) ? 0.0 : Double.NEGATIVE_INFINITY;
    }

    @Override
    public Map<String, DoubleTensor> dLogPdf(Double value) {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean isProbabilistic() {
        return false;
    }

    @Override
    public Double updateValue() {
        setValue(getDerivedValue());
        return getValue();
    }

    public abstract Double getDerivedValue();
}
