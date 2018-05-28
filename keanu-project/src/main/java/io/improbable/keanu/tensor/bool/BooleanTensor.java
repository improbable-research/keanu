package io.improbable.keanu.tensor.bool;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;

public interface BooleanTensor extends Tensor<Boolean> {

    static BooleanTensor create(boolean[] data, int[] shape) {
        return new SimpleBooleanTensor(data, shape);
    }

    static BooleanTensor scalar(boolean value) {
        return new SimpleBooleanTensor(value);
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

}
