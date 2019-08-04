package io.improbable.keanu.tensor.validate.policy;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.validate.TensorValueException;
import io.improbable.keanu.tensor.bool.BooleanTensor;

public class ThrowValueException<T, TENSOR extends Tensor<T, TENSOR>> implements TensorValidationPolicy<T, TENSOR> {
    private final String message;

    // package private - because it's created by the factory method TensorValidationPolicy.throwMessage
    ThrowValueException(String message) {
        this.message = message;
    }

    @Override
    public TENSOR handle(TENSOR tensor, BooleanTensor result) {
        if (!result.allTrue().scalar()) {
            throw new TensorValueException(message, result);
        }
        return tensor;
    }
}
