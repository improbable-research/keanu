package io.improbable.keanu.vertices.number;

import io.improbable.keanu.BaseFixedPointTensor;
import io.improbable.keanu.tensor.FixedPointTensor;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.LessThanOrEqualVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

import java.util.function.Function;

public interface FixedPointTensorVertex<T extends Number, TENSOR extends FixedPointTensor<T, TENSOR>, VERTEX extends FixedPointTensorVertex<T, TENSOR, VERTEX>>
    extends NumberTensorVertex<T, TENSOR, VERTEX>, BaseFixedPointTensor<BooleanVertex, IntegerVertex, DoubleVertex, T, VERTEX> {

    BooleanVertex toBoolean();

    DoubleVertex toDouble();

    IntegerVertex toInteger();

    VERTEX sum();

    VERTEX sum(int... overDimensions);

    VERTEX cumSum(int requestedDimension);

    VERTEX product();

    VERTEX product(int... overDimensions);

    VERTEX cumProd(int requestedDimension);

    VERTEX max();

    VERTEX max(VERTEX max);

    VERTEX min();

    VERTEX min(VERTEX min);

    VERTEX clamp(VERTEX min, VERTEX max);

    VERTEX matrixMultiply(VERTEX that);

    VERTEX tensorMultiply(VERTEX value, int[] dimLeft, int[] dimsRight);

    VERTEX abs();

    VERTEX minus(T value);

    VERTEX minus(VERTEX that);

    VERTEX reverseMinus(VERTEX value);

    VERTEX reverseMinus(T value);

    VERTEX plus(T value);

    VERTEX plus(VERTEX that);

    VERTEX unaryMinus();

    VERTEX times(T value);

    VERTEX times(VERTEX that);

    VERTEX div(T value);

    VERTEX div(VERTEX value);

    VERTEX reverseDiv(T value);

    VERTEX reverseDiv(VERTEX value);

    VERTEX pow(VERTEX exponent);

    VERTEX pow(T exponent);

    VERTEX average();

    VERTEX standardDeviation();

    IntegerVertex argMax(int axis);

    IntegerVertex argMax();

    IntegerVertex argMin(int axis);

    IntegerVertex argMin();

    VERTEX setWithMask(VERTEX mask, T value);

    VERTEX apply(Function<T, T> function);

    VERTEX safeLogTimes(VERTEX y);

    BooleanVertex equalsWithinEpsilon(VERTEX other, T epsilon);

    BooleanVertex lessThan(VERTEX value);

    default BooleanVertex lessThanOrEqual(VERTEX rhs) {
        return new LessThanOrEqualVertex<>(getThis(), rhs);
    }

    BooleanVertex greaterThan(VERTEX value);

    BooleanVertex greaterThanOrEqual(VERTEX value);

    BooleanVertex lessThan(T value);

    BooleanVertex lessThanOrEqual(T value);

    BooleanVertex greaterThan(T value);

    BooleanVertex greaterThanOrEqual(T value);

    VERTEX greaterThanMask(VERTEX greaterThanThis);

    VERTEX greaterThanOrEqualToMask(VERTEX greaterThanOrEqualThis);

    VERTEX lessThanMask(VERTEX lessThanThis);

    VERTEX lessThanOrEqualToMask(VERTEX lessThanOrEqualThis);

    VERTEX getThis();
}
