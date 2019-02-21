package io.improbable.keanu.tensor.bool;

import io.improbable.keanu.kotlin.BooleanOperators;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import org.apache.commons.lang3.ArrayUtils;

import static io.improbable.keanu.tensor.TensorShape.getAbsoluteDimension;

public interface BooleanTensor extends Tensor<Boolean>, BooleanOperators<BooleanTensor> {

    static BooleanTensor create(boolean value, long[] shape) {
        return new SimpleBooleanTensor(value, shape);
    }

    static BooleanTensor create(boolean[] values, long... shape) {
        return new SimpleBooleanTensor(values, shape);
    }

    static BooleanTensor create(boolean... values) {
        return create(values, values.length);
    }

    static BooleanTensor scalar(boolean scalarValue) {
        return new SimpleBooleanTensor(scalarValue);
    }

    static BooleanTensor trues(long... shape) {
        return new SimpleBooleanTensor(true, shape);
    }

    static BooleanTensor falses(long... shape) {
        return new SimpleBooleanTensor(false, shape);
    }

     /**
     * @param dimension  the dimension along which toStack are stacked
     * @param toStack    an array of BooleanTensor's of the same shape
     * @return  a BooleanTensor with toStack joined along a new dimension
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

    static BooleanTensor concat(int dimension, BooleanTensor[] toConcat) {
        DoubleTensor[] toDoubles = new DoubleTensor[toConcat.length];

        for (int i = 0; i < toConcat.length; i++) {
            toDoubles[i] = toConcat[i].toDoubleMask();
        }

        DoubleTensor concat = DoubleTensor.concat(dimension, toDoubles);
        double[] concatFlat = concat.asFlatDoubleArray();
        boolean[] data = new boolean[concat.asFlatDoubleArray().length];

        for (int i = 0; i < data.length; i++) {
            data[i] = concatFlat[i] == 1.0;
        }

        return new SimpleBooleanTensor(data, concat.getShape());
    }

    @Override
    BooleanTensor reshape(long... newShape);

    @Override
    BooleanTensor duplicate();

    BooleanTensor and(BooleanTensor that);

    default BooleanTensor and(boolean that) {
        return this.and(BooleanTensor.scalar(that));
    }

    BooleanTensor or(BooleanTensor that);

    default BooleanTensor or(boolean that) {
        return this.or(BooleanTensor.scalar(that));
    }

    BooleanTensor xor(BooleanTensor that);

    BooleanTensor not();

    DoubleTensor doubleWhere(DoubleTensor trueValue, DoubleTensor falseValue);

    IntegerTensor integerWhere(IntegerTensor trueValue, IntegerTensor falseValue);

    BooleanTensor booleanWhere(BooleanTensor trueValue, BooleanTensor falseValue);

    <T, TENSOR extends Tensor<T>> TENSOR where(TENSOR trueValue, TENSOR falseValue);

    BooleanTensor andInPlace(BooleanTensor that);

    BooleanTensor orInPlace(BooleanTensor that);

    BooleanTensor xorInPlace(BooleanTensor that);

    BooleanTensor notInPlace();

    boolean allTrue();

    boolean allFalse();

    DoubleTensor toDoubleMask();

    IntegerTensor toIntegerMask();

    @Override
    BooleanTensor slice(int dimension, long index);

    boolean[] asFlatBooleanArray();

}
