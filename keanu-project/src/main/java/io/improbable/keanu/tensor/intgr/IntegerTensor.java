package io.improbable.keanu.tensor.intgr;

import java.util.Arrays;
import java.util.function.Function;

import io.improbable.keanu.kotlin.IntegerOperators;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;

public interface IntegerTensor extends NumberTensor<Integer, IntegerTensor>, IntegerOperators<IntegerTensor> {

    IntegerTensor ZERO_SCALAR = scalar(0);

    IntegerTensor ONE_SCALAR = scalar(1);

    IntegerTensor TWO_SCALAR = scalar(2);

    static IntegerTensor create(int value, int[] shape) {
        if (Arrays.equals(shape, Tensor.SCALAR_SHAPE)) {
            return new ScalarIntegerTensor(value);
        } else {
            return Nd4jIntegerTensor.create(value, shape);
        }
    }

    static IntegerTensor create(int[] values, int... shape) {
        if (Arrays.equals(shape, Tensor.SCALAR_SHAPE) && values.length == 1) {
            return new ScalarIntegerTensor(values[0]);
        } else {
            return Nd4jIntegerTensor.create(values, shape);
        }
    }

    static IntegerTensor create(int... values) {
        return create(values, 1, values.length);
    }

    static IntegerTensor ones(int... shape) {
        if (Arrays.equals(shape, Tensor.SCALAR_SHAPE)) {
            return new ScalarIntegerTensor(1);
        } else {
            return Nd4jIntegerTensor.ones(shape);
        }
    }

    static IntegerTensor eye(int n) {
        if (n == 1) {
            return new ScalarIntegerTensor(1);
        } else {
            return Nd4jIntegerTensor.eye(n);
        }
    }

    static IntegerTensor zeros(int... shape) {
        if (Arrays.equals(shape, Tensor.SCALAR_SHAPE)) {
            return new ScalarIntegerTensor(0);
        } else {
            return Nd4jIntegerTensor.zeros(shape);
        }
    }

    static IntegerTensor scalar(int scalarValue) {
        return new ScalarIntegerTensor(scalarValue);
    }

    static IntegerTensor placeHolder(int[] shape) {
        return new ScalarIntegerTensor(shape);
    }

    @Override
    IntegerTensor setValue(Integer value, int... index);

    @Override
    IntegerTensor reshape(int... newShape);

    @Override
    IntegerTensor duplicate();

    IntegerTensor diag();

    IntegerTensor transpose();

    IntegerTensor sum(int... overDimensions);

    //New tensor Ops and transforms

    IntegerTensor minus(int value);

    IntegerTensor plus(int value);

    IntegerTensor times(int value);

    IntegerTensor div(int value);

    IntegerTensor pow(int exponent);

    IntegerTensor minus(IntegerTensor that);

    IntegerTensor plus(IntegerTensor that);

    IntegerTensor times(IntegerTensor that);

    IntegerTensor matrixMultiply(IntegerTensor value);

    IntegerTensor tensorMultiply(IntegerTensor value, int[] dimLeft, int[] dimsRight);

    IntegerTensor div(IntegerTensor that);

    IntegerTensor unaryMinus();

    IntegerTensor abs();

    IntegerTensor getGreaterThanMask(IntegerTensor greaterThanThis);

    IntegerTensor getGreaterThanOrEqualToMask(IntegerTensor greaterThanThis);

    IntegerTensor getLessThanMask(IntegerTensor lessThanThis);

    IntegerTensor getLessThanOrEqualToMask(IntegerTensor lessThanThis);

    IntegerTensor setWithMaskInPlace(IntegerTensor mask, Integer value);

    IntegerTensor setWithMask(IntegerTensor mask, Integer value);

    IntegerTensor apply(Function<Integer, Integer> function);

    @Override
    IntegerTensor slice(int dimension, int index);

    // In Place

    IntegerTensor minusInPlace(int value);

    IntegerTensor plusInPlace(int value);

    IntegerTensor timesInPlace(int value);

    IntegerTensor divInPlace(int value);

    IntegerTensor powInPlace(int exponent);

    // Comparisons

    BooleanTensor lessThan(int value);

    BooleanTensor lessThanOrEqual(int value);

    BooleanTensor greaterThan(int value);

    BooleanTensor greaterThanOrEqual(int value);

}
