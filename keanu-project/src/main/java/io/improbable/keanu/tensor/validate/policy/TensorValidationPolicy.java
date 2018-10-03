package io.improbable.keanu.tensor.validate.policy;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;

public interface TensorValidationPolicy<TENSOR extends Tensor<?>> {

    static <DATATYPE, TENSOR extends Tensor<DATATYPE>> TensorValidationPolicy<TENSOR> changeValueTo(
            DATATYPE v) {
        return new ChangeValueTo<>(v);
    }

    static <TENSOR extends Tensor<?>> TensorValidationPolicy<TENSOR> throwMessage(String message) {
        return new ThrowValueException<>(message);
    }

    TENSOR handle(TENSOR tensor, BooleanTensor result);
}
