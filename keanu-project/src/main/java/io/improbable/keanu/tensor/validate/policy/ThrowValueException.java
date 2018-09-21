package io.improbable.keanu.tensor.validate.policy;

import io.improbable.keanu.tensor.KeanuValueException;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;

public class ThrowValueException<TENSOR extends Tensor<?>> implements TensorValidationPolicy<TENSOR> {
    private final String message;

    // package private - because it's created by the factory method TensorValidationPolicy.throwMessage
    ThrowValueException(String message) {
        this.message = message;
    }

    @Override
    public TENSOR handle(TENSOR tensor, BooleanTensor result) {
        if (!result.allTrue()) {
            throw new KeanuValueException(message, result);
        }
        return tensor;
    }
}
