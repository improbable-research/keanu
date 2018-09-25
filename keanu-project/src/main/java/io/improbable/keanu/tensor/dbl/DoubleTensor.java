package io.improbable.keanu.tensor.dbl;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import io.improbable.keanu.kotlin.DoubleOperators;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;

public interface DoubleTensor extends NumberTensor<Double, DoubleTensor>, DoubleOperators<DoubleTensor> {

    DoubleTensor MINUS_ONE_SCALAR = scalar(-1.0);

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

    static DoubleTensor create(double... values) {
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

    static DoubleTensor zeros(int... shape) {
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

    static DoubleTensor placeHolder(int[] shape) {
        return new ScalarDoubleTensor(shape);
    }

    static DoubleTensor concat(int dimension, DoubleTensor... toConcat) {
        INDArray[] concatAsINDArray = new INDArray[toConcat.length];
        for (int i = 0; i < toConcat.length; i++) {
            concatAsINDArray[i] = Nd4jDoubleTensor.unsafeGetNd4J(toConcat[i]).dup();
        }
        INDArray concat = Nd4j.concat(dimension, concatAsINDArray);
        return new Nd4jDoubleTensor(concat);
    }

    @Override
    DoubleTensor setValue(Double value, int... index);

    @Override
    DoubleTensor reshape(int... newShape);

    DoubleTensor permute(int... rearrange);

    @Override
    DoubleTensor duplicate();

    DoubleTensor diag();

    DoubleTensor transpose();

    DoubleTensor sum(int... overDimensions);

    //New tensor Ops and transforms

    DoubleTensor reciprocal();

    DoubleTensor minus(double value);

    DoubleTensor plus(double value);

    DoubleTensor times(double value);

    DoubleTensor div(double value);

    DoubleTensor matrixMultiply(DoubleTensor value);

    DoubleTensor tensorMultiply(DoubleTensor value, int[] dimsLeft, int[] dimsRight);

    DoubleTensor pow(DoubleTensor exponent);

    DoubleTensor pow(double exponent);

    DoubleTensor sqrt();

    DoubleTensor log();

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

    DoubleTensor max(DoubleTensor max);

    DoubleTensor matrixInverse();

    double max();

    DoubleTensor min(DoubleTensor min);

    double min();

    double average();

    double standardDeviation();

    boolean equalsWithinEpsilon(DoubleTensor other, double epsilon);

    DoubleTensor standardize();

    DoubleTensor clamp(DoubleTensor min, DoubleTensor max);

    DoubleTensor ceil();

    DoubleTensor floor();

    DoubleTensor round();

    DoubleTensor sigmoid();

    DoubleTensor choleskyDecomposition();

    double determinant();

    double product();

    @Override
    DoubleTensor slice(int dimension, int index);

    List<DoubleTensor> split(int dimension, int[] splitAtIndices);

    default List<DoubleTensor> sliceAlongDimension(int dimension, int indexStart, int indexEnd) {
        List<DoubleTensor> slicedTensors = new ArrayList<>();

        for (int i = indexStart; i < indexEnd; i++) {
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

    DoubleTensor maxInPlace(DoubleTensor max);

    DoubleTensor minInPlace(DoubleTensor max);

    DoubleTensor clampInPlace(DoubleTensor min, DoubleTensor max);

    DoubleTensor ceilInPlace();

    DoubleTensor floorInPlace();

    DoubleTensor roundInPlace();

    DoubleTensor sigmoidInPlace();

    DoubleTensor standardizeInPlace();

    DoubleTensor setAllInPlace(double value);

    // Comparisons
    BooleanTensor lessThan(double value);

    BooleanTensor lessThanOrEqual(double value);

    BooleanTensor greaterThan(double value);

    BooleanTensor greaterThanOrEqual(double value);


}
