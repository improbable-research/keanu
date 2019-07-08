package io.improbable.keanu.vertices;

import io.improbable.keanu.BaseFixedPointTensor;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public interface FixedPointTensorVertex<T extends Number, TENSOR extends FixedPointTensorVertex<T, TENSOR>>
    extends NumberTensorVertex<T, TENSOR>, BaseFixedPointTensor<BooleanVertex, IntegerVertex, DoubleVertex, T, TENSOR> {
}
