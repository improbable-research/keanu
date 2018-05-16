package io.improbable.keanu.vertices.dbltensor;

import io.improbable.keanu.vertices.Vertex;

import java.util.Map;

public abstract class ContinuousTensorVertex<T> extends Vertex<T> {

    @Override
    public final double logProb(T value) {
        return logPdf(value);
    }

    public Map<Long, DoubleTensor> dLogProb(T value) {
        return dLogPdf(value);
    }

    public abstract double logPdf(T value);

    public abstract Map<Long, DoubleTensor> dLogPdf(T value);
}
