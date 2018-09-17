package io.improbable.keanu.vertices.model;

import io.improbable.keanu.vertices.VertexLabel;

public interface ModelResultProvider<T> {

    ModelVertex<T> getModel();

    VertexLabel getLabel();

    default T getValue() {
        if (!getModel().hasCalculated()) {
            getModel().calculate();
        }
        return getModel().getModelOutputValue(getLabel());
    }
}
