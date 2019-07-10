package io.improbable.keanu.tensor.bool;

import io.improbable.keanu.BaseBooleanTensor;
import io.improbable.keanu.kotlin.BooleanOperators;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import org.apache.commons.lang3.ArrayUtils;

import static io.improbable.keanu.tensor.TensorShape.getAbsoluteDimension;

public interface BooleanTensor extends
    Tensor<Boolean, BooleanTensor>,
    BooleanOperators<BooleanTensor>,
    BaseBooleanTensor<BooleanTensor, IntegerTensor, DoubleTensor> {

    static BooleanTensor create(boolean value, long[] shape) {
        return JVMBooleanTensor.create(value, shape);
    }

    static BooleanTensor create(boolean[] values, long... shape) {
        return JVMBooleanTensor.create(values, shape);
    }

    static BooleanTensor create(boolean... values) {
        return create(values, values.length);
    }

    static BooleanTensor scalar(boolean scalarValue) {
        return JVMBooleanTensor.scalar(scalarValue);
    }

    static BooleanTensor vector(boolean... values) {
        return create(values, values.length);
    }

    static BooleanTensor trues(long... shape) {
        return JVMBooleanTensor.create(true, shape);
    }

    static BooleanTensor falses(long... shape) {
        return JVMBooleanTensor.create(false, shape);
    }

    /**
     * @param dimension the dimension along which toStack are stacked
     * @param toStack   an array of BooleanTensor's of the same shape
     * @return a BooleanTensor with toStack joined along a new dimension
     * <p>
     * e.g. A, B, C = BooleanTensor.trues(4, 2)
     * <p>
     * BooleanTensor.stack(0, A, B, C) gives BooleanTensor.trues(3, 4, 2)
     * <p>
     * BooleanTensor.stack(1, A, B, C) gives BooleanTensor.trues(4, 3, 2)
     * <p>
     * BooleanTensor.stack(2, A, B, C) gives BooleanTensor.trues(4, 2, 3)
     * <p>
     * BooleanTensor.stack(-1, A, B, C) gives BooleanTensor.trues(4, 2, 3)
     */
    static BooleanTensor stack(int dimension, BooleanTensor... toStack) {
        long[] shape = toStack[0].getShape();
        int stackedRank = toStack[0].getRank() + 1;
        int absoluteDimension = getAbsoluteDimension(dimension, stackedRank);
        long[] stackedShape = ArrayUtils.insert(absoluteDimension, shape, 1);

        BooleanTensor[] reshaped = new BooleanTensor[toStack.length];
        for (int i = 0; i < toStack.length; i++) {
            reshaped[i] = toStack[i].reshape(stackedShape);
        }

        return concat(absoluteDimension, reshaped);
    }

    static BooleanTensor concat(int dimension, BooleanTensor... toConcat) {
        return JVMBooleanTensor.concat(dimension, toConcat);
    }

    default BooleanTensor and(BooleanTensor that) {
        return duplicate().andInPlace(that);
    }

    BooleanTensor andInPlace(BooleanTensor that);

    default BooleanTensor and(boolean that) {
        return duplicate().andInPlace(that);
    }

    BooleanTensor andInPlace(boolean that);

    default BooleanTensor or(BooleanTensor that) {
        return duplicate().orInPlace(that);
    }

    BooleanTensor orInPlace(BooleanTensor that);

    default BooleanTensor or(boolean that) {
        return duplicate().orInPlace(that);
    }

    BooleanTensor orInPlace(boolean that);

    default BooleanTensor xor(BooleanTensor that) {
        return duplicate().xorInPlace(that);
    }

    BooleanTensor xorInPlace(BooleanTensor that);

    default BooleanTensor not() {
        return duplicate().notInPlace();
    }

    BooleanTensor notInPlace();

    DoubleTensor doubleWhere(DoubleTensor trueValue, DoubleTensor falseValue);

    IntegerTensor integerWhere(IntegerTensor trueValue, IntegerTensor falseValue);

    BooleanTensor booleanWhere(BooleanTensor trueValue, BooleanTensor falseValue);

    <T, TENSOR extends Tensor<T, TENSOR>> TENSOR where(TENSOR trueValue, TENSOR falseValue);

    BooleanTensor allTrue();

    BooleanTensor allFalse();

    BooleanTensor anyTrue();

    BooleanTensor anyFalse();

    DoubleTensor toDoubleMask();

    IntegerTensor toIntegerMask();

    double[] asFlatDoubleArray();

    int[] asFlatIntegerArray();

    boolean[] asFlatBooleanArray();

}
