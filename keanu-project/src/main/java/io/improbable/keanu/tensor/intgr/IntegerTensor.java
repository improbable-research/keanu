package io.improbable.keanu.tensor.intgr;

import com.google.common.primitives.Ints;
import io.improbable.keanu.kotlin.IntegerOperators;
import io.improbable.keanu.tensor.FixedPointTensor;
import io.improbable.keanu.tensor.Tensor;
import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

import static io.improbable.keanu.tensor.TensorShape.getAbsoluteDimension;

public interface IntegerTensor extends FixedPointTensor<Integer, IntegerTensor>, IntegerOperators<IntegerTensor> {

    static IntegerTensor create(int value, long[] shape) {
        if (Arrays.equals(shape, Tensor.SCALAR_SHAPE)) {
            return new ScalarIntegerTensor(value);
        } else {
            return Nd4jIntegerTensor.create(value, shape);
        }
    }

    static IntegerTensor create(int[] values, long... shape) {
        if (Arrays.equals(shape, Tensor.SCALAR_SHAPE) && values.length == 1) {
            return new ScalarIntegerTensor(values[0]);
        } else {
            return Nd4jIntegerTensor.create(values, shape);
        }
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

    static IntegerTensor zeros(long... shape) {
        if (Arrays.equals(shape, Tensor.SCALAR_SHAPE)) {
            return new ScalarIntegerTensor(0);
        } else {
            return Nd4jIntegerTensor.zeros(shape);
        }
    }

    static IntegerTensor scalar(int scalarValue) {
        return new ScalarIntegerTensor(scalarValue);
    }

    static IntegerTensor vector(int... values) {
        return create(values, values.length);
    }

    /**
     * @param dimension the dimension along which toStack are stacked
     * @param toStack   an array of IntegerTensor
     * @return an IntegerTensor with toStack joined along a new dimension
     * <p>
     * e.g. A, B, C = IntegerTensor.ones(4, 2)
     * <p>
     * IntegerTensor.stack(0, A, B, C) gives IntegerTensor.ones(3, 4, 2)
     * <p>
     * IntegerTensor.stack(1, A, B, C) gives IntegerTensor.ones(4, 3, 2)
     * <p>
     * IntegerTensor.stack(2, A, B, C) gives IntegerTensor.ones(4, 2, 3)
     * <p>
     * IntegerTensor.stack(-1, A, B, C) gives IntegerTensor.ones(4, 2, 3)
     */
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

    /**
     * @param dimension the dimension along which the tensors will be joined
     * @param toConcat  an array of IntegerTensor
     * @return an IntegerTensor with toConcat joined along existing dimension
     * <p>
     * e.g. A, B, C = IntegerTensor.ones(4, 2)
     * <p>
     * IntegerTensor.concat(0, A, B, C) gives IntegerTensor.ones(12, 2)
     */
    static IntegerTensor concat(int dimension, IntegerTensor... toConcat) {
        INDArray[] concatAsINDArray = new INDArray[toConcat.length];
        for (int i = 0; i < toConcat.length; i++) {
            concatAsINDArray[i] = Nd4jIntegerTensor.unsafeGetNd4J(toConcat[i]).dup();
            if (concatAsINDArray[i].shape().length == 0) {
                concatAsINDArray[i] = concatAsINDArray[i].reshape(1);
            }
        }
        INDArray concat = Nd4j.concat(dimension, concatAsINDArray);
        return new Nd4jIntegerTensor(concat);
    }

    static IntegerTensor min(IntegerTensor a, IntegerTensor b) {
        return a.duplicate().minInPlace(b);
    }

    static IntegerTensor max(IntegerTensor a, IntegerTensor b) {
        return a.duplicate().maxInPlace(b);
    }

    default IntegerTensor reverseMinus(Integer that) {
        return this.unaryMinus().plus(that);
    }

    default IntegerTensor reverseDiv(Integer that) {
        return IntegerTensor.scalar(that).div(this);
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
