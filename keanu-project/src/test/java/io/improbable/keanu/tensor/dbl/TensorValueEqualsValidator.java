package io.improbable.keanu.tensor.dbl;

import io.improbable.keanu.tensor.bool.BooleanTensor;

public class TensorValueEqualsValidator implements TensorValidator {

    private final Double value;

    public TensorValueEqualsValidator(Double value) {

        this.value = value;
    }

    @Override
    public BooleanTensor check(DoubleTensor tensor) {
        DoubleTensor testTensor = DoubleTensor.create(value, tensor.getShape());
        return tensor.elementwiseEquals(testTensor);
    }
}
