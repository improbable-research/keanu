package io.improbable.keanu.vertices;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.update.ValueUpdater;

public abstract class DiscreteVertex<T extends Tensor> extends Vertex<T> {

    public DiscreteVertex(ValueUpdater<T> valueUpdater) {
        super(valueUpdater);
    }
}
