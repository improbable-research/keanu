package io.improbable.keanu.vertices.bool.probabilistic;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;
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

    default Map<Vertex, DoubleTensor> dLogPmf(boolean value, Set<Vertex> withRespectTo) {
        return dLogPmf(BooleanTensor.scalar(value), withRespectTo);
    }

    default Map<Vertex, DoubleTensor> dLogPmf(boolean[] values, Set<Vertex> withRespectTo) {
        return dLogPmf(BooleanTensor.create(values), withRespectTo);
    }

    default Map<Vertex, DoubleTensor> dLogPmf(BooleanTensor value, Set<Vertex> withRespectTo) {
        return dLogProb(value, withRespectTo);
    }
}
