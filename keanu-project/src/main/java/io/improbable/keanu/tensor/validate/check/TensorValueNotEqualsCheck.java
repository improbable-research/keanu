package io.improbable.keanu.tensor.validate.check;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;

public class TensorValueNotEqualsCheck<DATATYPE, TENSOR extends Tensor<DATATYPE>> implements TensorValueChecker<TENSOR> {

    private final DATATYPE value;

    public TensorValueNotEqualsCheck(DATATYPE value) {
        this.value = value;
    }

    @Override
    public BooleanTensor check(TENSOR tensor) {
        return tensor.elementwiseEquals(value).not();
    }
}
