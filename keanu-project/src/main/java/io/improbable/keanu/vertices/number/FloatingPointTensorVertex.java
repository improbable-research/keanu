package io.improbable.keanu.vertices.number;

import io.improbable.keanu.BaseFloatingPointTensor;
import io.improbable.keanu.tensor.FloatingPointTensor;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.NaNArgMaxVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.NaNArgMinVertex;

public interface FloatingPointTensorVertex<T extends Number, TENSOR extends FloatingPointTensor<T, TENSOR>, VERTEX extends FloatingPointTensorVertex<T, TENSOR, VERTEX>>
    extends NumberTensorVertex<T, TENSOR, VERTEX>, BaseFloatingPointTensor<BooleanVertex, IntegerVertex, DoubleVertex, T, VERTEX> {

    @Override
    default IntegerVertex nanArgMax(int axis) {
        return new NaNArgMaxVertex<>(this, axis);
    }

    @Override
    default IntegerVertex nanArgMax() {
        return new NaNArgMaxVertex<>(this);
    }

    @Override
    default IntegerVertex nanArgMin(int axis) {
        return new NaNArgMinVertex<>(this, axis);
    }

    @Override
    default IntegerVertex nanArgMin() {
        return new NaNArgMinVertex<>(this);
    }
}
