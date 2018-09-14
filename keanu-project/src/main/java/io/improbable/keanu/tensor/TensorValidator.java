package io.improbable.keanu.tensor;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

public interface TensorValidator {
    BooleanTensor check(DoubleTensor tensor);

    DoubleTensor validate(DoubleTensor containsZero);
}
