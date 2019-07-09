package io.improbable.keanu.vertices.number;

import io.improbable.keanu.BaseFloatingPointTensor;
import io.improbable.keanu.tensor.FloatingPointTensor;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public interface FloatingPointTensorVertex<T extends Number, TENSOR extends FloatingPointTensor<T, TENSOR>, VERTEX extends FloatingPointTensorVertex<T, TENSOR, VERTEX>>
    extends NumberTensorVertex<T, TENSOR, VERTEX>, BaseFloatingPointTensor<BooleanVertex, IntegerVertex, DoubleVertex, T, VERTEX> {

}
