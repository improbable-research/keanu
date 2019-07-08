package io.improbable.keanu.vertices;

import io.improbable.keanu.BaseNumberTensor;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public interface NumberTensorVertex<T extends Number, TENSOR extends NumberTensorVertex<T, TENSOR>>
    extends BaseNumberTensor<BooleanVertex, IntegerVertex, DoubleVertex, T, TENSOR> {

}
