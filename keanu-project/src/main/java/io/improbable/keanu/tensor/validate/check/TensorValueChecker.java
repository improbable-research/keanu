package io.improbable.keanu.tensor.validate.check;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;

public interface TensorValueChecker<TENSOR extends Tensor<?>> {
    BooleanTensor check(TENSOR tensor);
}
