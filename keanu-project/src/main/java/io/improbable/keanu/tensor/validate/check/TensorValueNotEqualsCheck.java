package io.improbable.keanu.tensor.validate.check;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;

public class TensorValueNotEqualsCheck<DATATYPE, TENSOR extends Tensor<DATATYPE>> implements TensorValueChecker<TENSOR> {

    private final DATATYPE value;

    public TensorValueNotEqualsCheck(DATATYPE value) {
        this.value = value;
    }

    @Override
    public BooleanTensor check(TENSOR tensor) {
        if (tensor instanceof DoubleTensor) {
            DoubleTensor testTensor = DoubleTensor.create((Double) value, tensor.getShape());
            return tensor.elementwiseEquals(testTensor).not();
        } else if (tensor instanceof IntegerTensor) {
            IntegerTensor testTensor = IntegerTensor.create((Integer) value, tensor.getShape());
            return tensor.elementwiseEquals(testTensor).not();
        } else {
            throw new ClassCastException("Cannot handle tensor of type " + tensor.getClass().getSimpleName());
        }
    }
}
