package io.improbable.keanu.tensor.intgr;

import com.google.common.primitives.Ints;
import io.improbable.keanu.kotlin.IntegerOperators;
import io.improbable.keanu.tensor.FixedPointTensor;
import io.improbable.keanu.tensor.TensorFactories;
import org.apache.commons.lang3.ArrayUtils;

import java.util.Arrays;

import static io.improbable.keanu.tensor.TensorShape.getAbsoluteDimension;

public interface IntegerTensor extends FixedPointTensor<Integer, IntegerTensor>, IntegerOperators<IntegerTensor> {

    static IntegerTensor create(long value, long[] shape) {
        return TensorFactories.integerTensorFactory.create(value, shape);
    }

    static IntegerTensor create(int[] values, long... shape) {
        return TensorFactories.integerTensorFactory.create(values, shape);
    }

    static IntegerTensor create(int... values) {
        return create(values, values.length);
    }

    /**
     * Creates an IntegerTensor from a long[]. Will throw an exception if any value in the long[] cannot be represented as an integer
     *
     * @param values long[] of values
     * @param shape  shape of the tensor
     * @return a new IntegerTensor
     */
    static IntegerTensor create(long[] values, long... shape) {
        int[] ints = Arrays.stream(values).mapToInt(Ints::checkedCast).toArray();
        return IntegerTensor.create(ints, shape);
    }

    /**
     * Creates an IntegerTensor from a long[]. Will throw an exception if any value in the long[] cannot be represented as an integer
     *
     * @param values long[] of values
     * @return a new IntegerTensor
     */
    static IntegerTensor create(long... values) {
        return IntegerTensor.create(values, values.length);
    }

    static IntegerTensor ones(long... shape) {
        return TensorFactories.integerTensorFactory.ones(shape);
    }

    static IntegerTensor eye(int n) {
        return TensorFactories.integerTensorFactory.eye(n);
    }

    static IntegerTensor zeros(long... shape) {
        return TensorFactories.integerTensorFactory.zeros(shape);
    }

    static IntegerTensor scalar(long scalarValue) {
        return TensorFactories.integerTensorFactory.scalar(scalarValue);
    }

    static IntegerTensor vector(int... values) {
        return create(values, values.length);
    }

    /**
     * @param start start of range
     * @param end   end of range (exclusive)
     * @return a vector of numbers from start incrementing by one to end (exclusively)
     */
    static IntegerTensor arange(int start, int end) {
        return TensorFactories.integerTensorFactory.arange(start, end);
    }

    static IntegerTensor arange(int end) {
        return TensorFactories.integerTensorFactory.arange(0, end);
    }

    /**
     * @param start    start of range
     * @param end      end of range (exclusive)
     * @param stepSize size of step from start to end
     * @return a vector of numbers starting at start and stepping to end (exclusively)
     */
    static IntegerTensor arange(int start, int end, int stepSize) {
        return TensorFactories.integerTensorFactory.arange(start, end, stepSize);
    }

    static IntegerTensor stack(int dimension, IntegerTensor... toStack) {
        long[] shape = toStack[0].getShape();
        int absoluteDimension = getAbsoluteDimension(dimension, shape.length + 1);
        long[] stackedShape = ArrayUtils.insert(absoluteDimension, shape, 1);

        IntegerTensor[] reshaped = new IntegerTensor[toStack.length];
        for (int i = 0; i < toStack.length; i++) {
            reshaped[i] = toStack[i].reshape(stackedShape);
        }

        return concat(absoluteDimension, reshaped);
    }

    static IntegerTensor concat(IntegerTensor... toConcat) {
        return concat(0, toConcat);
    }

    static IntegerTensor concat(int dimension, IntegerTensor... toConcat) {
        return TensorFactories.integerTensorFactory.concat(dimension, toConcat);
    }

    static IntegerTensor min(IntegerTensor a, IntegerTensor b) {
        return a.duplicate().minInPlace(b);
    }

    static IntegerTensor max(IntegerTensor a, IntegerTensor b) {
        return a.duplicate().maxInPlace(b);
    }

    // Kotlin unboxes to the primitive but does not match the Java
    @Override
    default IntegerTensor plus(int value) {
        return plus((Integer) value);
    }

    @Override
    default IntegerTensor minus(int value) {
        return minus((Integer) value);
    }

    @Override
    default IntegerTensor reverseMinus(int value) {
        return reverseMinus((Integer) value);
    }

    @Override
    default IntegerTensor times(int value) {
        return times((Integer) value);
    }

    @Override
    default IntegerTensor div(int value) {
        return div((Integer) value);
    }

    @Override
    default IntegerTensor reverseDiv(int value) {
        return reverseDiv((Integer) value);
    }

    @Override
    default IntegerTensor pow(int exponent) {
        return pow((Integer) exponent);
    }
}
