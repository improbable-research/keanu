package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.Probabilistic;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public interface ProbabilisticDouble extends Probabilistic<DoubleTensor> {
    default double logPdf(double value) {
        return logPdf(DoubleTensor.scalar(value));
    }

    default double logPdf(double[] values) {
        return logPdf(DoubleTensor.create(values));
    }

    default double logPdf(DoubleTensor value) {
        return logProb(value);
    }

    default Map<IVertex, DoubleTensor> dLogPdf(double value, Set<IVertex> withRespectTo) {
        return dLogPdf(DoubleTensor.scalar(value), withRespectTo);
    }

    default Map<IVertex, DoubleTensor> dLogPdf(double value,IVertex... withRespectTo) {
        return dLogPdf(DoubleTensor.scalar(value), new HashSet<>(Arrays.asList(withRespectTo)));
    }

    default Map<IVertex, DoubleTensor> dLogPdf(double[] values, Set<IVertex> withRespectTo) {
        return dLogPdf(DoubleTensor.create(values), withRespectTo);
    }

    default Map<IVertex, DoubleTensor> dLogPdf(double[] values,IVertex... withRespectTo) {
        return dLogPdf(DoubleTensor.create(values), new HashSet<>(Arrays.asList(withRespectTo)));
    }

    default Map<IVertex, DoubleTensor> dLogPdf(DoubleTensor value, Set<IVertex> withRespectTo) {
        return dLogProb(value, withRespectTo);
    }

    default Map<IVertex, DoubleTensor> dLogPdf(DoubleTensor value,IVertex... withRespectTo) {
        return dLogPdf(value, new HashSet<>(Arrays.asList(withRespectTo)));
    }
}
