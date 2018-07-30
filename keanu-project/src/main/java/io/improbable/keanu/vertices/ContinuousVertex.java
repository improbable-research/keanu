package io.improbable.keanu.vertices;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.Map;

public abstract class ContinuousVertex<T extends Tensor> extends Vertex<T> {

    @Override
    public final double logProb(T value) {
        return logPdf(value);
    }

    public Map<Long, DoubleTensor> dLogProb(T value) {
        return dLogPdf(value);
    }

    public abstract double logPdf(T value);

    public abstract Map<Long, DoubleTensor> dLogPdf(T value);

}
