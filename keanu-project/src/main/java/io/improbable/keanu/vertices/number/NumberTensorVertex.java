package io.improbable.keanu.vertices.number;

import io.improbable.keanu.BaseNumberTensor;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.TensorVertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public interface NumberTensorVertex<T extends Number, TENSOR extends Tensor<T, TENSOR>, VERTEX extends NumberTensorVertex<T, TENSOR, VERTEX>>
    extends TensorVertex<T, TENSOR, VERTEX>, BaseNumberTensor<BooleanVertex, IntegerVertex, DoubleVertex, T, VERTEX> {

}
