package io.improbable.keanu.tensor.dbl;

import io.improbable.keanu.kotlin.DoubleOperators;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static io.improbable.keanu.tensor.TensorShape.getAbsoluteDimension;


public interface DoubleTensor extends NumberTensor<Double, DoubleTensor>, DoubleOperators<DoubleTensor> {

    DoubleTensor MINUS_ONE_SCALAR = scalar(-1.0);

    DoubleTensor ZERO_SCALAR = scalar(0.0);

    DoubleTensor ONE_SCALAR = scalar(1.0);

    DoubleTensor TWO_SCALAR = scalar(2.0);

    static DoubleTensor create(double value, long[] shape) {
        if (Arrays.equals(shape, Tensor.SCALAR_SHAPE)) {
            return new ScalarDoubleTensor(value);
        } else {
            return Nd4jDoubleTensor.create(value, shape);
        }
    }

    static DoubleTensor create(double[] values, long... shape) {
        if (Arrays.equals(shape, Tensor.SCALAR_SHAPE) && values.length == 1) {
            return new ScalarDoubleTensor(values[0]);
        } else {
            return Nd4jDoubleTensor.create(values, shape);
        }
    }

    static DoubleTensor create(double... values) {
        return create(values, values.length);
    }

    static DoubleTensor ones(long... shape) {
        if (Arrays.equals(shape, Tensor.SCALAR_SHAPE)) {
            return new ScalarDoubleTensor(1.0);
        } else {
            return Nd4jDoubleTensor.ones(shape);
        }
    }

    static DoubleTensor eye(long n) {
        if (n == 1) {
            return new ScalarDoubleTensor(1.0);
        } else {
            return Nd4jDoubleTensor.eye(n);
        }
    }

    static DoubleTensor zeros(long... shape) {
        if (Arrays.equals(shape, Tensor.SCALAR_SHAPE)) {
            return new ScalarDoubleTensor(0.0);
        } else {
            return Nd4jDoubleTensor.zeros(shape);
        }
    }

    static DoubleTensor linspace(double start, double end, int numberOfPoints) {
        return Nd4jDoubleTensor.linspace(start, end, numberOfPoints);
    }

    /**
     * @param start start of range
     * @param end   end of range (exclusive)
     * @return a vector of numbers from start incrementing by one to end (exclusively)
     */
    static DoubleTensor arange(double start, double end) {
        return Nd4jDoubleTensor.arange(start, end);
    }

    /**
     * @param start    start of range
     * @param end      end of range (exclusive)
     * @param stepSize size of step from start to end
     * @return a vector of numbers starting at start and stepping to end (exclusively)
     */
    static DoubleTensor arange(double start, double end, double stepSize) {
        return Nd4jDoubleTensor.arange(start, end, stepSize);
    }

    static DoubleTensor scalar(double scalarValue) {
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
    static DoubleTensor stack(int dimension, DoubleTensor... toStack) {
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

    static DoubleTensor concat(DoubleTensor... toConcat) {
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
    static DoubleTensor concat(int dimension, DoubleTensor... toConcat) {
        INDArray[] concatAsINDArray = new INDArray[toConcat.length];
        for (int i = 0; i < toConcat.length; i++) {
            concatAsINDArray[i] = Nd4jDoubleTensor.unsafeGetNd4J(toConcat[i]).dup();
            if (concatAsINDArray[i].shape().length == 0) {
                concatAsINDArray[i] = concatAsINDArray[i].reshape(1);
            }
        }
        INDArray concat = Nd4j.concat(dimension, concatAsINDArray);
        return new Nd4jDoubleTensor(concat);
    }

    static DoubleTensor min(DoubleTensor a, DoubleTensor b) {
        return a.duplicate().minInPlace(b);
    }

    static DoubleTensor max(DoubleTensor a, DoubleTensor b) {
        return a.duplicate().maxInPlace(b);
    }

    @Override
    DoubleTensor setValue(Double value, long... index);

    @Override
    DoubleTensor reshape(long... newShape);

    DoubleTensor permute(int... rearrange);

    @Override
    DoubleTensor duplicate();

    DoubleTensor diag();

    DoubleTensor transpose();

    DoubleTensor sum(int... overDimensions);

    //New tensor Ops and transforms

    DoubleTensor reciprocal();

    DoubleTensor minus(double value);

    default DoubleTensor reverseMinus(double value) {
        return DoubleTensor.scalar(value).minus(this);
    }

    DoubleTensor plus(double value);

    DoubleTensor times(double value);

    DoubleTensor div(double value);

    default DoubleTensor reverseDiv(double value) {
        return DoubleTensor.scalar(value).div(this);
    }

    DoubleTensor matrixMultiply(DoubleTensor value);

    DoubleTensor tensorMultiply(DoubleTensor value, int[] dimsLeft, int[] dimsRight);

    DoubleTensor pow(DoubleTensor exponent);

    DoubleTensor pow(double exponent);

    DoubleTensor sqrt();

    DoubleTensor log();

    DoubleTensor safeLogTimes(DoubleTensor y);

    DoubleTensor logGamma();

    DoubleTensor digamma();

    DoubleTensor sin();

    DoubleTensor cos();

    DoubleTensor tan();

    DoubleTensor atan();

    DoubleTensor atan2(double y);

    DoubleTensor atan2(DoubleTensor y);

    DoubleTensor asin();

    DoubleTensor acos();

    DoubleTensor exp();

    DoubleTensor matrixInverse();

    double max();

    double min();

    double average();

    double standardDeviation();

    boolean equalsWithinEpsilon(DoubleTensor other, double epsilon);

    DoubleTensor standardize();

    DoubleTensor replaceNaN(double value);

    DoubleTensor clamp(DoubleTensor min, DoubleTensor max);

    DoubleTensor ceil();

    DoubleTensor floor();

    DoubleTensor round();

    DoubleTensor sigmoid();

    DoubleTensor choleskyDecomposition();

    double determinant();

    double product();

    @Override
    DoubleTensor slice(int dimension, long index);

    List<DoubleTensor> split(int dimension, long... splitAtIndices);

    default List<DoubleTensor> sliceAlongDimension(int dimension, long indexStart, long indexEnd) {
        List<DoubleTensor> slicedTensors = new ArrayList<>();

        for (long i = indexStart; i < indexEnd; i++) {
            slicedTensors.add(slice(dimension, i));
        }

        return slicedTensors;
    }

    //In place Ops and Transforms. These mutate the source vertex (i.e. this).

    DoubleTensor reciprocalInPlace();

    DoubleTensor minusInPlace(double value);

    DoubleTensor plusInPlace(double value);

    DoubleTensor timesInPlace(double value);

    DoubleTensor divInPlace(double value);

    DoubleTensor powInPlace(double exponent);

    DoubleTensor sqrtInPlace();

    DoubleTensor logInPlace();

    DoubleTensor safeLogTimesInPlace(DoubleTensor y);

    DoubleTensor logGammaInPlace();

    DoubleTensor digammaInPlace();

    DoubleTensor sinInPlace();

    DoubleTensor cosInPlace();

    DoubleTensor tanInPlace();

    DoubleTensor atanInPlace();

    DoubleTensor atan2InPlace(double y);

    DoubleTensor atan2InPlace(DoubleTensor y);

    DoubleTensor asinInPlace();

    DoubleTensor acosInPlace();

    DoubleTensor expInPlace();

    DoubleTensor minInPlace(DoubleTensor min);

    DoubleTensor maxInPlace(DoubleTensor max);

    DoubleTensor clampInPlace(DoubleTensor min, DoubleTensor max);

    DoubleTensor ceilInPlace();

    DoubleTensor floorInPlace();

    DoubleTensor roundInPlace();

    DoubleTensor sigmoidInPlace();

    DoubleTensor standardizeInPlace();

    DoubleTensor replaceNaNInPlace(double value);

    DoubleTensor setAllInPlace(double value);

    // Comparisons
    BooleanTensor lessThan(double value);

    BooleanTensor lessThanOrEqual(double value);

    BooleanTensor greaterThan(double value);

    BooleanTensor greaterThanOrEqual(double value);

    BooleanTensor notNaN();

    default BooleanTensor isNaN() {
        return notNaN().not();
    }

}
