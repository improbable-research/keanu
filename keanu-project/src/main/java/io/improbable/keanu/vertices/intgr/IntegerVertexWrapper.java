package io.improbable.keanu.vertices.intgr;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.tensor.VertexWrapper;

public class IntegerVertexWrapper extends VertexWrapper<IntegerTensor, IntegerVertex> implements IntegerVertex {

    public static IntegerVertex wrapIfNeeded(Vertex<IntegerTensor, ?> vertex) {
        if (vertex instanceof IntegerVertex) {
            return (IntegerVertex) vertex;
        }

        if (vertex instanceof NonProbabilisticVertex) {
            return new IntegerVertexWrapper((NonProbabilisticVertex<IntegerTensor, ?>) vertex);
        } else {
            throw new IllegalStateException();
        }
    }

    public IntegerVertexWrapper(NonProbabilisticVertex<IntegerTensor, ?> vertex) {
        super(vertex);
    }
}
