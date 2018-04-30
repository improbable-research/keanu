package io.improbable.keanu.vertices;

import java.util.Map;

public abstract class ContinuousVertex<T> extends Vertex<T> {

    @Override
    public final double logProb(T value) {
        return logPdf(value);
    }

    @Override
    public final Map<String, Double> dLogProb(T value) {
        return dLogPdf(value);
    }

    public abstract double logPdf(T value);

    public abstract Map<String, Double> dLogPdf(T value);

}
