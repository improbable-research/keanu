package io.improbable.keanu.vertices;

import io.improbable.keanu.BaseFloatingPointTensor;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public interface FloatingPointTensorVertex<T extends Number, TENSOR extends FloatingPointTensorVertex<T, TENSOR>>
    extends NumberTensorVertex<T, TENSOR>, BaseFloatingPointTensor<BooleanVertex, IntegerVertex, DoubleVertex, T, TENSOR> {

}
