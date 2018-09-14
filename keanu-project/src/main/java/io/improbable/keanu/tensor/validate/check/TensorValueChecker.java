package io.improbable.keanu.tensor.validate.check;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

public interface TensorValueChecker {
    BooleanTensor check(DoubleTensor tensor);
}
