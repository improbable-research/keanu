package io.improbable.keanu.tensor.validate.check;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;

public interface TensorValueChecker<T extends Tensor<?>> {
    BooleanTensor check(T tensor);
}
