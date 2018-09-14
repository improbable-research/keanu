package io.improbable.keanu.tensor;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.validate.TensorValidationPolicy;

public class TensorValueEqualsValidator implements TensorValidator {

    private final Double value;
    private TensorValidationPolicy policy;

    public TensorValueEqualsValidator(Double value) {
        this.value = value;
    }

    @Override
    public BooleanTensor check(DoubleTensor tensor) {
        DoubleTensor testTensor = DoubleTensor.create(value, tensor.getShape());
        return tensor.elementwiseEquals(testTensor);
    }

    @Override
    public DoubleTensor validate(DoubleTensor tensor) {
        BooleanTensor result = check(tensor);
        return policy.handle(tensor, result);
    }

    public TensorValidator withPolicy(TensorValidationPolicy policy) {
        this.policy = policy;
        return this;
    }
}
