package io.improbable.keanu.tensor.dbl;

import io.improbable.keanu.tensor.bool.BooleanTensor;

public class TensorValidator {

    private final Double value;

    public TensorValidator(Double value) {

        this.value = value;
    }

    public BooleanTensor check(DoubleTensor tensor) {
        DoubleTensor testTensor = DoubleTensor.create(value, tensor.getShape());
        return tensor.elementwiseEquals(testTensor);
    }
}
