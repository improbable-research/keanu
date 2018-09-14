package io.improbable.keanu.tensor.validate.check;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

public class TensorValueNotEqualsCheck implements TensorValueChecker {

    private final Double value;

    public TensorValueNotEqualsCheck(Double value) {
        this.value = value;
    }

    @Override
    public BooleanTensor check(DoubleTensor tensor) {
        DoubleTensor testTensor = DoubleTensor.create(value, tensor.getShape());
        return tensor.elementwiseEquals(testTensor);
    }
}
