package io.improbable.keanu.tensor.validate.policy;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;

public interface TensorValidationPolicy<T extends Tensor<?>> {

    static <U, T extends Tensor<U>> TensorValidationPolicy<T> changeValueTo(U v) {
        return new ChangeValueTo<U, T>(v);
    }

    static <T extends Tensor<?>> TensorValidationPolicy<T> throwMessage(String message) {
        return new ThrowValueException<T>(message);
    }

    T handle(T tensor, BooleanTensor result);
}
