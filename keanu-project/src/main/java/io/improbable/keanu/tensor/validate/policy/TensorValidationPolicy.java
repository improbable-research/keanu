package io.improbable.keanu.tensor.validate.policy;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;

public interface TensorValidationPolicy<T, TENSOR extends Tensor<T, TENSOR>> {

    static <DATATYPE, TENSOR extends Tensor<DATATYPE, TENSOR>> TensorValidationPolicy<DATATYPE, TENSOR> changeValueTo(DATATYPE v) {
        return new ChangeValueTo<>(v);
    }

    static <DATATYPE, TENSOR extends Tensor<DATATYPE, TENSOR>> TensorValidationPolicy<DATATYPE, TENSOR> throwMessage(String message) {
        return new ThrowValueException<>(message);
    }

    TENSOR handle(TENSOR tensor, BooleanTensor result);
}
