package io.improbable.keanu.vertices.number;

import io.improbable.keanu.BaseFixedPointTensor;
import io.improbable.keanu.tensor.FixedPointTensor;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public interface FixedPointTensorVertex<T extends Number, TENSOR extends FixedPointTensor<T, TENSOR>, VERTEX extends FixedPointTensorVertex<T, TENSOR, VERTEX>>
    extends NumberTensorVertex<T, TENSOR, VERTEX>, BaseFixedPointTensor<BooleanVertex, IntegerVertex, DoubleVertex, T, VERTEX> {

}
