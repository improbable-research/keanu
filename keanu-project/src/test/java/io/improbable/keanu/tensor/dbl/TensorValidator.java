package io.improbable.keanu.tensor.dbl;

import io.improbable.keanu.tensor.bool.BooleanTensor;

public interface TensorValidator {
    BooleanTensor check(DoubleTensor tensor);
}
