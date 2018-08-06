package io.improbable.keanu.vertices.intgr.probabilistic;

import java.util.Map;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.Probabilistic;

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

    default Map<Long, DoubleTensor> dLogPmf(int value) {
        return dLogPmf(IntegerTensor.scalar(value));
    }

    default Map<Long, DoubleTensor> dLogPmf(int[] values) {
        return dLogPmf(IntegerTensor.create(values));
    }

    default Map<Long,DoubleTensor> dLogPmf(IntegerTensor value) {
        return dLogProb(value);
    }
}
