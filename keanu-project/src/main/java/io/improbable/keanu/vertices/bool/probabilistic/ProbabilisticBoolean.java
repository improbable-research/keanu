package io.improbable.keanu.vertices.bool.probabilistic;

import java.util.Map;
import java.util.Set;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;

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

    default Map<VertexId, DoubleTensor> dLogPmf(boolean value, Set<Vertex> withRespectTo) {
        return dLogPmf(BooleanTensor.scalar(value), withRespectTo);
    }

    default Map<VertexId, DoubleTensor> dLogPmf(boolean[] values, Set<Vertex> withRespectTo) {
        return dLogPmf(BooleanTensor.create(values), withRespectTo);
    }

    default Map<VertexId, DoubleTensor> dLogPmf(BooleanTensor value, Set<Vertex> withRespectTo) {
        return dLogProb(value, withRespectTo);
    }
}
