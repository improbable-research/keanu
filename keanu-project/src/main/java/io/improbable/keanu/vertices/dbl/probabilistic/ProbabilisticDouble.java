package io.improbable.keanu.vertices.dbl.probabilistic;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;

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

    default Map<Vertex, DoubleTensor> dLogPdf(double value, Set<Vertex> withRespectTo) {
        return dLogPdf(DoubleTensor.scalar(value), withRespectTo);
    }

    default Map<Vertex, DoubleTensor> dLogPdf(double value, Vertex... withRespectTo) {
        return dLogPdf(DoubleTensor.scalar(value), new HashSet<>(Arrays.asList(withRespectTo)));
    }

    default Map<Vertex, DoubleTensor> dLogPdf(double[] values, Set<Vertex> withRespectTo) {
        return dLogPdf(DoubleTensor.create(values), withRespectTo);
    }

    default Map<Vertex, DoubleTensor> dLogPdf(double[] values, Vertex... withRespectTo) {
        return dLogPdf(DoubleTensor.create(values), new HashSet<>(Arrays.asList(withRespectTo)));
    }

    default Map<Vertex, DoubleTensor> dLogPdf(DoubleTensor value, Set<Vertex> withRespectTo) {
        if (withRespectTo.isEmpty()) {
            throw new IllegalArgumentException("Must take dLogPdf wrt something");
        }
        return dLogProb(value, withRespectTo);
    }

    default Map<Vertex, DoubleTensor> dLogPdf(DoubleTensor value, Vertex... withRespectTo) {
        return dLogPdf(value, new HashSet<>(Arrays.asList(withRespectTo)));
    }
}
