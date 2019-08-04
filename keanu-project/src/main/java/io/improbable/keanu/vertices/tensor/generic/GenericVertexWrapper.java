package io.improbable.keanu.vertices.tensor.generic;

import io.improbable.keanu.tensor.generic.GenericTensor;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.tensor.VertexWrapper;

public class GenericVertexWrapper<T> extends VertexWrapper<GenericTensor<T>, GenericTensorVertex<T>> implements GenericTensorVertex<T> {

    public GenericVertexWrapper(NonProbabilisticVertex<GenericTensor<T>, GenericTensorVertex<T>> vertex) {
        super(vertex);
    }
}
