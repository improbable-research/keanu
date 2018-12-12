package io.improbable.keanu.tensor.intgr;

import static io.improbable.keanu.tensor.TensorShape.getAbsoluteDimension;

import io.improbable.keanu.kotlin.IntegerOperators;
import io.improbable.keanu.tensor.INDArrayShim;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.function.Function;

public interface IntegerTensor extends NumberTensor<Integer, IntegerTensor>, IntegerOperators<IntegerTensor> {

    IntegerTensor ZERO_SCALAR = scalar(0);

    IntegerTensor ONE_SCALAR = scalar(1);

    IntegerTensor TWO_SCALAR = scalar(2);

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

    /**
     * @param dimension  the dimension along which toStack are stacked
     * @param toStack    an array of IntegerTensor
     * @return  an IntegerTensor with toStack joined along a new dimension
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
     * @return  an IntegerTensor with toConcat joined along existing dimension
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

    @Override
    IntegerTensor setValue(Integer value, long... index);

    @Override
    IntegerTensor reshape(long... newShape);

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

    default IntegerTensor reverseMinus(int that) {
        return this.unaryMinus().plus(that);
    }

    IntegerTensor plus(IntegerTensor that);

    IntegerTensor times(IntegerTensor that);

    IntegerTensor matrixMultiply(IntegerTensor value);

    IntegerTensor tensorMultiply(IntegerTensor value, int[] dimLeft, int[] dimsRight);

    IntegerTensor div(IntegerTensor that);

    default IntegerTensor reverseDiv(int that) {
        return IntegerTensor.scalar(that).div(this);
    }

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
    IntegerTensor slice(int dimension, long index);

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

    IntegerTensor minInPlace(IntegerTensor min);

    IntegerTensor maxInPlace(IntegerTensor max);

    int min();

    int max();

}
