package io.improbable.keanu.vertices.tensor.number.fixed;

import io.improbable.keanu.BaseFixedPointTensor;
import io.improbable.keanu.tensor.FixedPointTensor;
import io.improbable.keanu.vertices.tensor.bool.BooleanVertex;
import io.improbable.keanu.vertices.tensor.number.NumberTensorVertex;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.IntegerVertex;
import io.improbable.keanu.vertices.tensor.number.fixed.operators.unary.ModVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;

public interface FixedPointTensorVertex<T extends Number, TENSOR extends FixedPointTensor<T, TENSOR>, VERTEX extends FixedPointTensorVertex<T, TENSOR, VERTEX>>
    extends NumberTensorVertex<T, TENSOR, VERTEX>, BaseFixedPointTensor<BooleanVertex, IntegerVertex, DoubleVertex, T, VERTEX> {

    default VERTEX mod(VERTEX that) {
        return wrap(new ModVertex<>(this, that));
    }
}
