package io.improbable.keanu.vertices;

import io.improbable.keanu.BaseTensor;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.bool.BooleanVertex;

public interface TensorVertex<T, TENSOR extends Tensor<T, TENSOR>, VERTEX extends BaseTensor<BooleanVertex, T, VERTEX>>
    extends Vertex<TENSOR>, BaseTensor<BooleanVertex, T, VERTEX> {

}
