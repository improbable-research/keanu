package io.improbable.keanu.vertices;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.update.ValueUpdater;

public abstract class ContinuousVertex<T extends Tensor> extends Vertex<T> {

    public ContinuousVertex(ValueUpdater<T> valueUpdater) {
        super(valueUpdater);
    }

}
