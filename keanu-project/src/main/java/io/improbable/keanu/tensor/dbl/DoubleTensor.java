package io.improbable.keanu.tensor.dbl;

import io.improbable.keanu.kotlin.DoubleOperators;
import io.improbable.keanu.tensor.FloatingPointTensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import org.apache.commons.lang3.ArrayUtils;

import java.util.ArrayList;
import java.util.List;

import static io.improbable.keanu.tensor.TensorShape.getAbsoluteDimension;


public abstract class DoubleTensor implements FloatingPointTensor<Double, DoubleTensor>, DoubleOperators<DoubleTensor> {

    private static DoubleTensorFactory factory = new JVMDoubleTensorFactory();

    public static void setFactory(DoubleTensorFactory factory) {
        DoubleTensor.factory = factory;
    }

    public static DoubleTensor create(double value, long[] shape) {
        return factory.create(value, shape);

    }

    public static DoubleTensor create(double[] values, long... shape) {
        return factory.create(values, shape);
    }

    public static DoubleTensor create(double... values) {
        return create(values, values.length);
    }

    public static DoubleTensor ones(long... shape) {
        return factory.ones(shape);
    }

    public static DoubleTensor eye(long n) {
        return factory.eye(n);
    }

    public static DoubleTensor zeros(long... shape) {
        return factory.zeros(shape);
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
        return factory.scalar(scalarValue);
    }

    public static DoubleTensor vector(double... values) {
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
    public DoubleTensor reverseMinus(Double value) {
        return DoubleTensor.scalar(value).minus(this);
    }

    @Override
    public DoubleTensor reverseDiv(Double value) {
        return DoubleTensor.scalar(value).div(this);
    }

    public abstract List<DoubleTensor> split(int dimension, long... splitAtIndices);

    public List<DoubleTensor> sliceAlongDimension(int dimension, long indexStart, long indexEnd) {
        List<DoubleTensor> slicedTensors = new ArrayList<>();

        for (long i = indexStart; i < indexEnd; i++) {
            slicedTensors.add(slice(dimension, i));
        }

        return slicedTensors;
    }

    //In place Ops and Transforms. These mutate the source vertex (i.e. this).

    public abstract DoubleTensor minusInPlace(Double value);

    public abstract DoubleTensor plusInPlace(Double value);

    public abstract DoubleTensor timesInPlace(Double value);

    public abstract DoubleTensor divInPlace(Double value);

    public abstract DoubleTensor powInPlace(Double exponent);

    public abstract DoubleTensor atan2InPlace(Double y);

    public abstract DoubleTensor replaceNaNInPlace(Double value);

    public abstract DoubleTensor setAllInPlace(Double value);

    public abstract DoubleTensor safeLogTimesInPlace(DoubleTensor y);

    public abstract DoubleTensor reciprocalInPlace();

    public abstract DoubleTensor sqrtInPlace();

    public abstract DoubleTensor logInPlace();

    public abstract DoubleTensor logGammaInPlace();

    public abstract DoubleTensor digammaInPlace();

    public abstract DoubleTensor sinInPlace();

    public abstract DoubleTensor cosInPlace();

    public abstract DoubleTensor tanInPlace();

    public abstract DoubleTensor atanInPlace();

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

    // Comparisons

    public abstract BooleanTensor lessThan(Double value);

    public abstract BooleanTensor lessThanOrEqual(Double value);

    public abstract BooleanTensor greaterThan(Double value);

    public abstract BooleanTensor greaterThanOrEqual(Double value);

    public abstract BooleanTensor notNaN();

    public BooleanTensor isNaN() {
        return notNaN().not();
    }

    // Kotlin unboxes to the primitive but does not match the Java
    @Override
    public DoubleTensor plus(double value) {
        return plus((Double) value);
    }

    @Override
    public DoubleTensor minus(double value) {
        return minus((Double) value);
    }

    @Override
    public DoubleTensor reverseMinus(double value) {
        return reverseMinus((Double) value);
    }

    @Override
    public DoubleTensor times(double value) {
        return times((Double) value);
    }

    @Override
    public DoubleTensor div(double value) {
        return div((Double) value);
    }

    @Override
    public DoubleTensor reverseDiv(double value) {
        return reverseDiv((Double) value);
    }

    @Override
    public DoubleTensor pow(double exponent) {
        return pow((Double) exponent);
    }

}
