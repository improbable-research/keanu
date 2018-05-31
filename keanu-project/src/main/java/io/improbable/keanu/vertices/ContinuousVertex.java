package io.improbable.keanu.vertices;

import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.Map;

public abstract class ContinuousVertex<T> extends Vertex<T> {

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
    @Override
    public final Map<Long, DoubleTensor> dLogProb(T value) {
        return dLogPdf(value);
    }

    public abstract double logPdf(T value);

    public abstract Map<Long, DoubleTensor> dLogPdf(T value);

}
