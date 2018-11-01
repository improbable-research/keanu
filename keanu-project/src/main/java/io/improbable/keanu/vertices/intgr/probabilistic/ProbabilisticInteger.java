package io.improbable.keanu.vertices.intgr.probabilistic;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;

import java.util.Map;
import java.util.Set;

public interface ProbabilisticInteger extends Probabilistic<IntegerTensor> {
    default double logPmf(int value) {
        return logPmf(IntegerTensor.scalar(value));
    }

    default double logPmf(int[] values) {
        return logPmf(IntegerTensor.create(values));
    }

    default double logPmf(IntegerTensor value) {
        return logProb(value);
    }

    default Map<Vertex, DoubleTensor> dLogPmf(int value, Set<Vertex> withRespectTo) {
        return dLogPmf(IntegerTensor.scalar(value), withRespectTo);
    }

    default Map<Vertex, DoubleTensor> dLogPmf(int[] values, Set<Vertex> withRespectTo) {
        return dLogPmf(IntegerTensor.create(values), withRespectTo);
    }

    default Map<Vertex, DoubleTensor> dLogPmf(IntegerTensor value, Set<Vertex> withRespectTo) {
        return dLogProb(value, withRespectTo);
    }

}
