package io.improbable.keanu.tensor.dbl;

import io.improbable.keanu.tensor.FloatingPointTensor;
import io.improbable.keanu.tensor.TensorFactories;
import org.apache.commons.lang3.ArrayUtils;

import static io.improbable.keanu.tensor.TensorShape.getAbsoluteDimension;

public interface DoubleTensor extends FloatingPointTensor<Double, DoubleTensor> {

    static DoubleTensor create(double value, long[] shape) {
        return TensorFactories.doubleTensorFactory.create(value, shape);
    }

    static DoubleTensor create(double[] values, long... shape) {
        return TensorFactories.doubleTensorFactory.create(values, shape);
    }

    static DoubleTensor create(double... values) {
        return create(values, values.length);
    }

    static DoubleTensor ones(long... shape) {
        return TensorFactories.doubleTensorFactory.ones(shape);
    }

    static DoubleTensor eye(long n) {
        return TensorFactories.doubleTensorFactory.eye(n);
    }

    static DoubleTensor zeros(long... shape) {
        return TensorFactories.doubleTensorFactory.zeros(shape);
    }

    static DoubleTensor linspace(double start, double end, int numberOfPoints) {
        return TensorFactories.doubleTensorFactory.linspace(start, end, numberOfPoints);
    }

    /**
     * @param start start of range
     * @param end   end of range (exclusive)
     * @return a vector of numbers from start incrementing by one to end (exclusively)
     */
    static DoubleTensor arange(double start, double end) {
        return TensorFactories.doubleTensorFactory.arange(start, end);
    }

    static DoubleTensor arange(double end) {
        return TensorFactories.doubleTensorFactory.arange(0, end);
    }

    /**
     * @param start    start of range
     * @param end      end of range (exclusive)
     * @param stepSize size of step from start to end
     * @return a vector of numbers starting at start and stepping to end (exclusively)
     */
    static DoubleTensor arange(double start, double end, double stepSize) {
        return TensorFactories.doubleTensorFactory.arange(start, end, stepSize);
    }

    static DoubleTensor scalar(double scalarValue) {
        return TensorFactories.doubleTensorFactory.scalar(scalarValue);
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
        return TensorFactories.doubleTensorFactory.concat(dimension, toConcat);
    }

    static DoubleTensor min(DoubleTensor a, DoubleTensor b) {
        return a.duplicate().minInPlace(b);
    }

    static DoubleTensor max(DoubleTensor a, DoubleTensor b) {
        return a.duplicate().maxInPlace(b);
    }

    // Kotlin unboxes to the primitive but does not match the Java
    @Override
    default DoubleTensor plus(double value) {
        return plus((Double) value);
    }

    @Override
    default DoubleTensor minus(double value) {
        return minus((Double) value);
    }

    @Override
    default DoubleTensor reverseMinus(double value) {
        return reverseMinus((Double) value);
    }

    @Override
    default DoubleTensor times(double value) {
        return times((Double) value);
    }

    @Override
    default DoubleTensor div(double value) {
        return div((Double) value);
    }

    @Override
    default DoubleTensor reverseDiv(double value) {
        return reverseDiv((Double) value);
    }

    @Override
    default DoubleTensor pow(double exponent) {
        return pow((Double) exponent);
    }

}
