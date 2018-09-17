package io.improbable.keanu.tensor.validate.policy;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;

public interface TensorValidationPolicy<TENSOR extends Tensor<?>> {

    static <DATATYPE, TENSOR extends Tensor<DATATYPE>> TensorValidationPolicy<TENSOR> changeValueTo(DATATYPE v) {
        return new ChangeValueTo<DATATYPE, TENSOR>(v);
    }

    static <TENSOR extends Tensor<?>> TensorValidationPolicy<TENSOR> throwMessage(String message) {
        return new ThrowValueException<TENSOR>(message);
    }

    void handle(TENSOR tensor, BooleanTensor result);
}
