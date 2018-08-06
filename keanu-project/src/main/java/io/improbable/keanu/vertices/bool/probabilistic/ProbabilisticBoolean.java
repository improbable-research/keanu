package io.improbable.keanu.vertices.bool.probabilistic;

import java.util.Map;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Probabilistic;

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

    default Map<Long, DoubleTensor> dLogPmf(boolean value) {
        return dLogPmf(BooleanTensor.scalar(value));
    }

    default Map<Long, DoubleTensor> dLogPmf(boolean[] values) {
        return dLogPmf(BooleanTensor.create(values));
    }

    default Map<Long,DoubleTensor> dLogPmf(BooleanTensor value) {
        return dLogProb(value);
    }
}
