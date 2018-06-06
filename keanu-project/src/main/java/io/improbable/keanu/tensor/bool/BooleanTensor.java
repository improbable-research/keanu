package io.improbable.keanu.tensor.bool;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;

public interface BooleanTensor extends Tensor<Boolean> {

    static BooleanTensor create(boolean value, int[] shape) {
        return new SimpleBooleanTensor(value, shape);
    }

    static BooleanTensor create(boolean[] values, int[] shape) {
        return new SimpleBooleanTensor(values, shape);
    }

    static BooleanTensor create(boolean[] values) {
        return create(values, new int[]{1, values.length});
    }

    static BooleanTensor scalar(boolean scalarValue) {
        return new SimpleBooleanTensor(scalarValue);
    }

    static BooleanTensor placeHolder(int[] shape) {
        return new SimpleBooleanTensor(shape);
    }

    BooleanTensor and(BooleanTensor that);

    BooleanTensor or(BooleanTensor that);

    BooleanTensor not();

    DoubleTensor setDoubleIf(DoubleTensor trueValue, DoubleTensor falseValue);

    IntegerTensor setIntegerIf(IntegerTensor trueValue, IntegerTensor falseValue);

    BooleanTensor setBooleanIf(BooleanTensor trueValue, BooleanTensor falseValue);

    <T> Tensor<T> setIf(Tensor<T> trueValue, Tensor<T> falseValue);

    BooleanTensor andInPlace(BooleanTensor that);

    BooleanTensor orInPlace(BooleanTensor that);

    BooleanTensor notInPlace();

    boolean allTrue();

    boolean allFalse();

    DoubleTensor toDoubleMask();

    IntegerTensor toIntegerMask();

}
