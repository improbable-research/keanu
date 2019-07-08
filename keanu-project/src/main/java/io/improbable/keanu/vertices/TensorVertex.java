package io.improbable.keanu.vertices;

import io.improbable.keanu.BaseTensor;
import io.improbable.keanu.vertices.bool.BooleanVertex;

public interface TensorVertex<T, TENSOR extends BaseTensor<BooleanVertex, T, TENSOR>>
    extends BaseTensor<BooleanVertex, T, TENSOR> {

}
