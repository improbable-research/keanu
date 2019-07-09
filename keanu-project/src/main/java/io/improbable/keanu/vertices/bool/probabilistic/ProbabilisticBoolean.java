package io.improbable.keanu.vertices.bool.probabilistic;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.Probabilistic;

import java.util.Map;
import java.util.Set;

public interface ProbabilisticBoolean extends Probabilistic<BooleanTensor> {

    default double logPmf(boolean value) {
        return logPmf(BooleanTensor.scalar(value));
    }

    default double logPmf(boolean[] values) {
        return logPmf(BooleanTensor.create(values));
    }

    default double logPmf(BooleanTensor value) {
        return logProb(value);
    }

    default Map<IVertex, DoubleTensor> dLogPmf(boolean value, Set<IVertex> withRespectTo) {
        return dLogPmf(BooleanTensor.scalar(value), withRespectTo);
    }

    default Map<IVertex, DoubleTensor> dLogPmf(boolean[] values, Set<IVertex> withRespectTo) {
        return dLogPmf(BooleanTensor.create(values), withRespectTo);
    }

    default Map<IVertex, DoubleTensor> dLogPmf(BooleanTensor value, Set<IVertex> withRespectTo) {
        return dLogProb(value, withRespectTo);
    }
}
