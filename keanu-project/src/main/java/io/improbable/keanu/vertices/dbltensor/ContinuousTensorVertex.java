package io.improbable.keanu.vertices.dbltensor;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.Vertex;

import java.util.Map;

public abstract class ContinuousTensorVertex<T extends Tensor> extends Vertex<T> {

    @Override
    public final double logProb(T value) {
        return logPdf(value);
    }

    public Map<Long, DoubleTensor> dLogProb(T value) {
        return dLogPdf(value);
    }

    public abstract double logPdf(T value);

    public abstract Map<Long, DoubleTensor> dLogPdf(T value);

    /**
     * TODO: Move this up to the Vertex base class when everything has been tensorized
     * @return true there is a tensor present and that tensor is more than just a shape placeholder
     */
    @Override
    public boolean hasValue() {
        return getRawValue() != null && !getRawValue().isShapePlaceholder();
    }

    public int[] getShape(){
        return getRawValue().getShape();
    }
}
