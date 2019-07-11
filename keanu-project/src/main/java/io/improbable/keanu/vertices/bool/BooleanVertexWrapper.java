package io.improbable.keanu.vertices.bool;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.tensor.VertexWrapper;

public class BooleanVertexWrapper extends VertexWrapper<BooleanTensor, BooleanVertex> implements BooleanVertex {

    public BooleanVertexWrapper(NonProbabilisticVertex<BooleanTensor, BooleanVertex> vertex) {
        super(vertex);
    }
}
