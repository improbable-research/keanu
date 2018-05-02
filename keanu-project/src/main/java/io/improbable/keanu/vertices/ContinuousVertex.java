package io.improbable.keanu.vertices;

import java.util.Map;

public abstract class ContinuousVertex<T> extends Vertex<T> implements LogProbDifferentiable<T> {

    @Override
    public final double logProb(T value) {
        return logPdf(value);
    }

    /**
     * The partial derivatives of the natural log prob.
     *
     * @param value at a given value
     * @return the partial derivatives of the log density
     */
    public final Map<String, Double> dLogProb(T value) {
        return dLogPdf(value);
    }

    public final Map<String, Double> dLogProbAtValue() {
        return dLogProb(getValue());
    }

    public abstract double logPdf(T value);

    public abstract Map<String, Double> dLogPdf(T value);

}
