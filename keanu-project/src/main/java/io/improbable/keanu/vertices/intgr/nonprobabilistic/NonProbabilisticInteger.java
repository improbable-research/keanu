package io.improbable.keanu.vertices.intgr.nonprobabilistic;

import io.improbable.keanu.vertices.NonProbabilisticObservationException;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

import java.util.Map;

public abstract class NonProbabilisticInteger extends IntegerVertex {

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
    public void observe(Integer value) {
        throw new NonProbabilisticObservationException();
    }

    @Override
    public double density(Integer value) {
        return this.getDerivedValue().equals(value) ? 1 : 0;
    }

    @Override
    public Map<String, Double> dDensityAtValue() {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean isProbabilistic() {
        return false;
    }

    @Override
    public Integer updateValue() {
        if (!this.isObserved()) {
            setValue(getDerivedValue());
        }
        return getValue();
    }

    public abstract Integer getDerivedValue();

}
