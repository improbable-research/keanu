package io.improbable.keanu.tensor.validate.policy;

import io.improbable.keanu.tensor.KeanuValueException;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

public class ThrowValueException implements TensorValidationPolicy {
    private final String message;

    public ThrowValueException(String message) {
        this.message = message;
    }

    @Override
    public DoubleTensor handle(DoubleTensor tensor, BooleanTensor result) {
        if (result.allFalse()) {
            return tensor;
        } else {
            throw new KeanuValueException(message, result);
        }
    }
}
