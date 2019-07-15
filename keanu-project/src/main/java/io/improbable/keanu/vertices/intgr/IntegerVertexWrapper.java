package io.improbable.keanu.vertices.intgr;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.tensor.VertexWrapper;

public class IntegerVertexWrapper extends VertexWrapper<IntegerTensor, IntegerVertex> implements IntegerVertex {

    public IntegerVertexWrapper(NonProbabilisticVertex<IntegerTensor, ?> vertex) {
        super(vertex);
    }
}
