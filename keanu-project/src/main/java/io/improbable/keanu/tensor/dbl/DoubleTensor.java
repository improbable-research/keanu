package io.improbable.keanu.tensor.dbl;

import java.util.Arrays;

import io.improbable.keanu.kotlin.DoubleOperators;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;

public interface DoubleTensor extends NumberTensor<Double, DoubleTensor>, DoubleOperators<DoubleTensor> {

    DoubleTensor ZERO_SCALAR = scalar(0.0);

    DoubleTensor ONE_SCALAR = scalar(1.0);

    DoubleTensor TWO_SCALAR = scalar(2.0);

    static DoubleTensor create(double value, int[] shape) {
        if (Arrays.equals(shape, Tensor.SCALAR_SHAPE)) {
            return new ScalarDoubleTensor(value);
        } else {
            return Nd4jDoubleTensor.create(value, shape);
        }
    }

    static DoubleTensor create(double[] values, int... shape) {
        if (Arrays.equals(shape, Tensor.SCALAR_SHAPE) && values.length == 1) {
            return new ScalarDoubleTensor(values[0]);
        } else {
            return Nd4jDoubleTensor.create(values, shape);
        }
    }

    static DoubleTensor create(double[] values) {
        return create(values, 1, values.length);
    }

    static DoubleTensor ones(int... shape) {
        if (Arrays.equals(shape, Tensor.SCALAR_SHAPE)) {
            return new ScalarDoubleTensor(1.0);
        } else {
            return Nd4jDoubleTensor.ones(shape);
        }
    }

    static DoubleTensor eye(int n) {
        if (n == 1) {
            return new ScalarDoubleTensor(1.0);
        } else {
            return Nd4jDoubleTensor.eye(n);
        }
    }

    static DoubleTensor zeros(int[] shape) {
        if (Arrays.equals(shape, Tensor.SCALAR_SHAPE)) {
            return new ScalarDoubleTensor(0.0);
        } else {
            return Nd4jDoubleTensor.zeros(shape);
        }
    }

    static DoubleTensor scalar(double scalarValue) {
        return new ScalarDoubleTensor(scalarValue);
    }

    static DoubleTensor placeHolder(int[] shape) {
        return new ScalarDoubleTensor(shape);
    }

    //New tensor Ops and transforms

    DoubleTensor reciprocal();

    DoubleTensor minus(double value);

    DoubleTensor plus(double value);

    DoubleTensor times(double value);

    DoubleTensor div(double value);

    DoubleTensor pow(double exponent);

    DoubleTensor sqrt();

    DoubleTensor tan();

    DoubleTensor atan();

    DoubleTensor atan2(double y);

    DoubleTensor atan2(DoubleTensor y);

    DoubleTensor max(DoubleTensor max);

    DoubleTensor inverse();

    double max();

    DoubleTensor min(DoubleTensor min);

    double min();

    double average();

    double standardDeviation();

    DoubleTensor standardize();

    DoubleTensor clamp(DoubleTensor min, DoubleTensor max);

    DoubleTensor ceil();

    DoubleTensor floor();

    DoubleTensor round();

    DoubleTensor sigmoid();

    DoubleTensor choleskyDecomposition();

    double determinant();

    DoubleTensor concat(int dimension, DoubleTensor... those);

    //In place Ops and Transforms. These mutate the source vertex (i.e. this).

    DoubleTensor reciprocalInPlace();

    DoubleTensor minusInPlace(double value);

    DoubleTensor plusInPlace(double value);

    DoubleTensor timesInPlace(double value);

    DoubleTensor divInPlace(double value);

    DoubleTensor powInPlace(double exponent);

    DoubleTensor sqrtInPlace();

    DoubleTensor logInPlace();

    DoubleTensor sinInPlace();

    DoubleTensor cosInPlace();

    DoubleTensor tanInPlace();

    DoubleTensor atanInPlace();

    DoubleTensor atan2InPlace(double y);

    DoubleTensor atan2InPlace(DoubleTensor y);

    DoubleTensor asinInPlace();

    DoubleTensor acosInPlace();

    DoubleTensor expInPlace();

    DoubleTensor maxInPlace(DoubleTensor max);

    DoubleTensor minInPlace(DoubleTensor max);

    DoubleTensor clampInPlace(DoubleTensor min, DoubleTensor max);

    DoubleTensor ceilInPlace();

    DoubleTensor floorInPlace();

    DoubleTensor roundInPlace();

    DoubleTensor sigmoidInPlace();

    DoubleTensor standardizeInPlace();

    // Comparisons
    BooleanTensor lessThan(double value);

    BooleanTensor lessThanOrEqual(double value);

    BooleanTensor greaterThan(double value);

    BooleanTensor greaterThanOrEqual(double value);


}
