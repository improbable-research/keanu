package io.improbable.keanu.vertices.dbl.probabilistic;

import java.util.Map;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Probabilistic;

public interface ProbabilisticDouble extends Probabilistic<DoubleTensor> {
    default double logProb(double value) {
        return logProb(DoubleTensor.scalar(value));
    }

    default double logProb(double[] values) {
        return logProb(DoubleTensor.create(values));
    }

    default Map<Long, DoubleTensor> dLogProb(double value) {
        return dLogProb(DoubleTensor.scalar(value));
    }

    default Map<Long, DoubleTensor> dLogProb(double[] values) {
        return dLogProb(DoubleTensor.create(values));
    }
}
