package io.improbable.keanu.tensor.dbl;

import io.improbable.keanu.kotlin.DoubleOperators;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import org.apache.commons.lang3.ArrayUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static io.improbable.keanu.tensor.TensorShape.getAbsoluteDimension;


public abstract class DoubleTensor implements NumberTensor<Double, DoubleTensor>, DoubleOperators<DoubleTensor> {

    private static DoubleTensorFactory factory = new JVMDoubleTensorFactory();

    public static void setFactory(DoubleTensorFactory factory) {
        DoubleTensor.factory = factory;
    }

    public final static DoubleTensor MINUS_ONE_SCALAR = scalar(-1.0);

    public final static DoubleTensor ZERO_SCALAR = scalar(0.0);

    public final static DoubleTensor ONE_SCALAR = scalar(1.0);

    public final static DoubleTensor TWO_SCALAR = scalar(2.0);

    public static DoubleTensor create(double value, long[] shape) {
        if (Arrays.equals(shape, Tensor.SCALAR_SHAPE)) {
            return new ScalarDoubleTensor(value);
        } else {
            return factory.create(value, shape);
        }
    }

    public static DoubleTensor create(double[] values, long... shape) {
        if (Arrays.equals(shape, Tensor.SCALAR_SHAPE) && values.length == 1) {
            return new ScalarDoubleTensor(values[0]);
        } else {
            return factory.create(values, shape);
        }
    }

    public static DoubleTensor create(double... values) {
        return create(values, values.length);
    }

    public static DoubleTensor ones(long... shape) {
        if (Arrays.equals(shape, Tensor.SCALAR_SHAPE)) {
            return new ScalarDoubleTensor(1.0);
        } else {
            return factory.ones(shape);
        }
    }

    public static DoubleTensor eye(long n) {
        if (n == 1) {
            return new ScalarDoubleTensor(1.0);
        } else {
            return factory.eye(n);
        }
    }

    public static DoubleTensor zeros(long... shape) {
        if (Arrays.equals(shape, Tensor.SCALAR_SHAPE)) {
            return new ScalarDoubleTensor(0.0);
        } else {
            return factory.zeros(shape);
        }
    }

    public static DoubleTensor linspace(double start, double end, int numberOfPoints) {
        return factory.linspace(start, end, numberOfPoints);
    }

    /**
     * @param start start of range
     * @param end   end of range (exclusive)
     * @return a vector of numbers from start incrementing by one to end (exclusively)
     */
    public static DoubleTensor arange(double start, double end) {
        return factory.arange(start, end);
    }

    /**
     * @param start    start of range
     * @param end      end of range (exclusive)
     * @param stepSize size of step from start to end
     * @return a vector of numbers starting at start and stepping to end (exclusively)
     */
    public static DoubleTensor arange(double start, double end, double stepSize) {
        return factory.arange(start, end, stepSize);
    }

    public static DoubleTensor scalar(double scalarValue) {
        return new ScalarDoubleTensor(scalarValue);
    }

    static DoubleTensor vector(double... values) {
        return DoubleTensor.create(values, values.length);
    }

    /**
     * @param dimension the dimension along which toStack are stacked
     * @param toStack   an array of DoubleTensor's of the same shape
     * @return a DoubleTensor with toStack joined along a new dimension
     * <p>
     * e.g. A, B, C = DoubleTensor.ones(4, 2)
     * <p>
     * DoubleTensor.stack(0, A, B, C) gives DoubleTensor.ones(3, 4, 2)
     * <p>
     * DoubleTensor.stack(1, A, B, C) gives DoubleTensor.ones(4, 3, 2)
     * <p>
     * DoubleTensor.stack(2, A, B, C) gives DoubleTensor.ones(4, 2, 3)
     * <p>
     * DoubleTensor.stack(-1, A, B, C) gives DoubleTensor.ones(4, 2, 3)
     */
    public static DoubleTensor stack(int dimension, DoubleTensor... toStack) {
        long[] shape = toStack[0].getShape();
        int stackedRank = toStack[0].getRank() + 1;
        int absoluteDimension = getAbsoluteDimension(dimension, stackedRank);
        long[] stackedShape = ArrayUtils.insert(absoluteDimension, shape, 1);

        DoubleTensor[] reshaped = new DoubleTensor[toStack.length];
        for (int i = 0; i < toStack.length; i++) {
            reshaped[i] = toStack[i].reshape(stackedShape);
        }

        return concat(absoluteDimension, reshaped);
    }

    public static DoubleTensor concat(DoubleTensor... toConcat) {
        return concat(0, toConcat);
    }

    /**
     * @param dimension the dimension along which the tensors will be joined
     * @param toConcat  an array of DoubleTensor
     * @return a DoubleTensor with toConcat joined along an existing dimension
     * <p>
     * e.g. A, B, C = DoubleTensor.ones(4, 2)
     * <p>
     * DoubleTensor.concat(0, A, B, C) gives DoubleTensor.ones(12, 2)
     */
    public static DoubleTensor concat(int dimension, DoubleTensor... toConcat) {
        return factory.concat(dimension, toConcat);
    }

    public static DoubleTensor min(DoubleTensor a, DoubleTensor b) {
        return a.duplicate().minInPlace(b);
    }

    public static DoubleTensor max(DoubleTensor a, DoubleTensor b) {
        return a.duplicate().maxInPlace(b);
    }

    @Override
    public abstract DoubleTensor setValue(Double value, long... index);

    @Override
    public abstract DoubleTensor reshape(long... newShape);

    public abstract DoubleTensor permute(int... rearrange);

    @Override
    public abstract DoubleTensor duplicate();

    public abstract DoubleTensor diag();

    public abstract DoubleTensor transpose();

    public abstract DoubleTensor sum(int... overDimensions);

    //New tensor Ops and transforms

    public abstract DoubleTensor reciprocal();

    public abstract DoubleTensor minus(double value);

    public DoubleTensor reverseMinus(double value) {
        return DoubleTensor.scalar(value).minus(this);
    }

    public abstract DoubleTensor plus(double value);

    public abstract DoubleTensor times(double value);

    public abstract DoubleTensor div(double value);

    public DoubleTensor reverseDiv(double value) {
        return DoubleTensor.scalar(value).div(this);
    }

    public abstract DoubleTensor matrixMultiply(DoubleTensor value);

    public abstract DoubleTensor tensorMultiply(DoubleTensor value, int[] dimsLeft, int[] dimsRight);

    public abstract DoubleTensor pow(DoubleTensor exponent);

    public abstract DoubleTensor pow(double exponent);

    public abstract DoubleTensor sqrt();

    public abstract DoubleTensor log();

    public abstract DoubleTensor safeLogTimes(DoubleTensor y);

    public abstract DoubleTensor logGamma();

    public abstract DoubleTensor digamma();

    public abstract DoubleTensor sin();

    public abstract DoubleTensor cos();

    public abstract DoubleTensor tan();

    public abstract DoubleTensor atan();

    public abstract DoubleTensor atan2(double y);

    public abstract DoubleTensor atan2(DoubleTensor y);

    public abstract DoubleTensor asin();

    public abstract DoubleTensor acos();

    public abstract DoubleTensor exp();

    public abstract DoubleTensor matrixInverse();

    public abstract double max();

    public abstract double min();

    public abstract double average();

    public abstract double standardDeviation();

    public abstract boolean equalsWithinEpsilon(DoubleTensor other, double epsilon);

    public abstract DoubleTensor standardize();

    public abstract DoubleTensor replaceNaN(double value);

    public abstract DoubleTensor clamp(DoubleTensor min, DoubleTensor max);

    public abstract DoubleTensor ceil();

    public abstract DoubleTensor floor();

    /**
     * @return The tensor with the elements rounded half up
     * e.g. 1.5 is 2
     * e.g. -2.5 is -3
     */
    public abstract DoubleTensor round();

    public abstract DoubleTensor sigmoid();

    public abstract DoubleTensor choleskyDecomposition();

    public abstract double determinant();

    public abstract double product();

    @Override
    public abstract DoubleTensor slice(int dimension, long index);

    public abstract List<DoubleTensor> split(int dimension, long... splitAtIndices);

    public List<DoubleTensor> sliceAlongDimension(int dimension, long indexStart, long indexEnd) {
        List<DoubleTensor> slicedTensors = new ArrayList<>();

        for (long i = indexStart; i < indexEnd; i++) {
            slicedTensors.add(slice(dimension, i));
        }

        return slicedTensors;
    }

    //In place Ops and Transforms. These mutate the source vertex (i.e. this).

    public abstract DoubleTensor reciprocalInPlace();

    public abstract DoubleTensor minusInPlace(double value);

    public abstract DoubleTensor plusInPlace(double value);

    public abstract DoubleTensor timesInPlace(double value);

    public abstract DoubleTensor divInPlace(double value);

    public abstract DoubleTensor powInPlace(double exponent);

    public abstract DoubleTensor sqrtInPlace();

    public abstract DoubleTensor logInPlace();

    public abstract DoubleTensor safeLogTimesInPlace(DoubleTensor y);

    public abstract DoubleTensor logGammaInPlace();

    public abstract DoubleTensor digammaInPlace();

    public abstract DoubleTensor sinInPlace();

    public abstract DoubleTensor cosInPlace();

    public abstract DoubleTensor tanInPlace();

    public abstract DoubleTensor atanInPlace();

    public abstract DoubleTensor atan2InPlace(double y);

    public abstract DoubleTensor atan2InPlace(DoubleTensor y);

    public abstract DoubleTensor asinInPlace();

    public abstract DoubleTensor acosInPlace();

    public abstract DoubleTensor expInPlace();

    public abstract DoubleTensor minInPlace(DoubleTensor min);

    public abstract DoubleTensor maxInPlace(DoubleTensor max);

    public abstract DoubleTensor clampInPlace(DoubleTensor min, DoubleTensor max);

    public abstract DoubleTensor ceilInPlace();

    public abstract DoubleTensor floorInPlace();

    public abstract DoubleTensor roundInPlace();

    public abstract DoubleTensor sigmoidInPlace();

    public abstract DoubleTensor standardizeInPlace();

    public abstract DoubleTensor replaceNaNInPlace(double value);

    public abstract DoubleTensor setAllInPlace(double value);

    // Comparisons
    public abstract BooleanTensor lessThan(double value);

    public abstract BooleanTensor lessThanOrEqual(double value);

    public abstract BooleanTensor greaterThan(double value);

    public abstract BooleanTensor greaterThanOrEqual(double value);

    public abstract BooleanTensor notNaN();

    public BooleanTensor isNaN() {
        return notNaN().not();
    }

}
