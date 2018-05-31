package io.improbable.keanu.tensor.intgr;

import io.improbable.keanu.kotlin.IntegerOperators;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;

import java.util.function.Function;

public interface IntegerTensor extends NumberTensor<Integer>, IntegerOperators<IntegerTensor> {

    static IntegerTensor scalar(int scalarValue) {
        return new ScalarIntegerTensor(scalarValue);
    }

    static IntegerTensor create(int[] values, int[] shape) {
        return Nd4jIntegerTensor.create(values, shape);
    }

    static IntegerTensor create(double value, int[] shape) {
        return Nd4jIntegerTensor.create(value, shape);
    }

    static IntegerTensor ones(int[] shape) {
        return Nd4jIntegerTensor.ones(shape);
    }

    static IntegerTensor zeros(int[] shape) {
        return Nd4jIntegerTensor.zeros(shape);
    }

    static IntegerTensor placeHolder(int[] shape) {
        return new Nd4jIntegerTensor(shape);
    }

    //New tensor Ops and transforms

    IntegerTensor minus(int value);

    IntegerTensor plus(int value);

    IntegerTensor times(int value);

    IntegerTensor div(int value);

    IntegerTensor pow(IntegerTensor exponent);

    IntegerTensor pow(int exponent);

    IntegerTensor minus(IntegerTensor that);

    IntegerTensor plus(IntegerTensor that);

    IntegerTensor times(IntegerTensor that);

    IntegerTensor div(IntegerTensor that);

    IntegerTensor unaryMinus();

    IntegerTensor abs();

    IntegerTensor getGreaterThanMask(IntegerTensor greaterThanThis);

    IntegerTensor getGreaterThanOrEqualToMask(IntegerTensor greaterThanThis);

    IntegerTensor getLessThanMask(IntegerTensor lessThanThis);

    IntegerTensor getLessThanOrEqualToMask(IntegerTensor lessThanThis);

    IntegerTensor setWithMaskInPlace(IntegerTensor mask, int value);

    IntegerTensor setWithMask(IntegerTensor mask, int value);

    IntegerTensor apply(Function<Integer, Integer> function);

    // In Place

    IntegerTensor minusInPlace(int value);

    IntegerTensor plusInPlace(int value);

    IntegerTensor timesInPlace(int value);

    IntegerTensor divInPlace(int value);

    IntegerTensor powInPlace(IntegerTensor exponent);

    IntegerTensor powInPlace(int exponent);

    IntegerTensor minusInPlace(IntegerTensor that);

    IntegerTensor plusInPlace(IntegerTensor that);

    IntegerTensor timesInPlace(IntegerTensor that);

    IntegerTensor divInPlace(IntegerTensor that);

    IntegerTensor unaryMinusInPlace();

    IntegerTensor absInPlace();

    IntegerTensor applyInPlace(Function<Integer, Integer> function);

    // Comparisons
    BooleanTensor lessThan(int value);

    BooleanTensor lessThanOrEqual(int value);

    BooleanTensor lessThan(IntegerTensor value);

    BooleanTensor lessThanOrEqual(IntegerTensor value);

    BooleanTensor greaterThan(int value);

    BooleanTensor greaterThanOrEqual(int value);

    BooleanTensor greaterThan(IntegerTensor value);

    BooleanTensor greaterThanOrEqual(IntegerTensor value);

}
