package io.improbable.keanu.tensor.validate.check;

import java.util.function.Function;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;

public class CustomTensorValueChecker<TENSOR extends Tensor<?>> implements TensorValueChecker<TENSOR> {
    private final Function<TENSOR, BooleanTensor> checkFunction;

    public CustomTensorValueChecker(Function<TENSOR, BooleanTensor> checkFunction) {
        this.checkFunction = checkFunction;
    }

    @Override
    public BooleanTensor check(TENSOR tensor) {
        return checkFunction.apply(tensor);
    }
}
