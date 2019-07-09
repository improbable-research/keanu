package io.improbable.keanu.vertices;

import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public interface Probabilistic<T> extends Observable<T>, Samplable<T> {

    /**
     * This is the natural log of the probability at the supplied value. In the
     * case of continuous vertices, this is actually the log of the density, which
     * will differ from the probability by a constant.
     *
     * @param value The supplied value.
     * @return The natural log of the probability function at the supplied value.
     * For continuous variables this is called the PDF (probability density function).
     * For discrete variables this is called the PMF (probability mass function).
     */
    double logProb(T value);

    /**
     * The partial derivatives of the natural log prob.
     *
     * @param atValue       at a given value
     * @param withRespectTo list of parents to differentiate with respect to
     * @return the partial derivatives of the log of the probability function at the supplied value.
     * For continuous variables this is called the PDF (probability density function).
     * For discrete variables this is called the PMF (probability mass function).
     */
    Map<IVertex, DoubleTensor> dLogProb(T atValue, Set<? extends IVertex> withRespectTo);

    default Map<IVertex, DoubleTensor> dLogProb(T atValue, IVertex... withRespectTo) {
        return dLogProb(atValue, new HashSet<>(Arrays.asList(withRespectTo)));
    }

    T getValue();

    default double logProbAtValue() {
        return logProb(getValue());
    }

    default Map<IVertex, DoubleTensor> dLogProbAtValue(Set<? extends IVertex> withRespectTo) {
        return dLogProb(getValue(), withRespectTo);
    }

    default Map<IVertex, DoubleTensor> dLogProbAtValue(IVertex... withRespectTo) {
        return dLogProb(getValue(), withRespectTo);
    }

}
