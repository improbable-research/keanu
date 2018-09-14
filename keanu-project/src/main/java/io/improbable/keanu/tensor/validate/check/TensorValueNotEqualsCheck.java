package io.improbable.keanu.tensor.validate.check;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;

public class TensorValueNotEqualsCheck<U, T extends Tensor<U>> implements TensorValueChecker<T> {

    private final U value;

    public TensorValueNotEqualsCheck(U value) {
        this.value = value;
    }

    @Override
    public BooleanTensor check(T tensor) {
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
