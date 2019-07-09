package io.improbable.keanu.vertices.intgr.probabilistic;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.Probabilistic;

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

    default Map<IVertex, DoubleTensor> dLogPmf(int value, Set<IVertex> withRespectTo) {
        return dLogPmf(IntegerTensor.scalar(value), withRespectTo);
    }

    default Map<IVertex, DoubleTensor> dLogPmf(int[] values, Set<IVertex> withRespectTo) {
        return dLogPmf(IntegerTensor.create(values), withRespectTo);
    }

    default Map<IVertex, DoubleTensor> dLogPmf(IntegerTensor value, Set<IVertex> withRespectTo) {
        return dLogProb(value, withRespectTo);
    }

}
