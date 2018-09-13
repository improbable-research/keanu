package io.improbable.keanu.tensor;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

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
