package io.improbable.keanu.vertices;

import io.improbable.keanu.vertices.dbltensor.DoubleTensor;

import java.util.Map;

public abstract class ContinuousVertex<T> extends Vertex<T> {

    @Override
    public final double logProb(T value) {
        return logPdf(value);
    }

    @Override
    public final Map<String, DoubleTensor> dLogProb(T value) {
        return dLogPdf(value);
    }

    public abstract double logPdf(T value);

    public abstract Map<String, DoubleTensor> dLogPdf(T value);

}
