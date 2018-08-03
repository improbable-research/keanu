package io.improbable.keanu.vertices;

import java.util.List;
import java.util.Map;

import com.google.common.collect.ImmutableList;

import io.improbable.keanu.tensor.dbl.DoubleTensor;

public interface Probabilistic<T> extends Observable<T> {

    /**
     * This is the natural log of the probability at the supplied value. In the
     * case of continuous vertices, this is actually the log of the density, which
     * will differ from the probability by a constant.
     *
     * @param value The supplied value.
     * @return The natural log of the probability density at the supplied value
     */
    double logProb(T value);

    /**
     * The partial derivatives of the natural log prob.
     *
     * @param value at a given value
     * @return the partial derivatives of the log density
     */
     Map<Long, DoubleTensor> dLogProb(T value);

    T getValue();
    void setValue(T value);

    default double logProbAtValue() {
        return logProb(getValue());
    };

    default Map<Long, DoubleTensor> dLogProbAtValue() {
        return dLogProb(getValue());
    }

    static <V extends Vertex & Probabilistic> List<V> filter(Iterable<? extends Vertex> vertices) {
        ImmutableList.Builder<V> probabilisticVertices = ImmutableList.builder();
        for (Vertex v : vertices) {
            if (v instanceof Probabilistic) {
                probabilisticVertices.add((V) v);
            }
        }
        return probabilisticVertices.build();
    }
}
