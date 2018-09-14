package io.improbable.keanu.tensor.validate.policy;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

public interface TensorValidationPolicy {
    static TensorValidationPolicy changeValueTo(double v) {
        return new ChangeValueTo(v);
    }

    DoubleTensor handle(DoubleTensor tensor, BooleanTensor result);
}
