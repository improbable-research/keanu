package io.improbable.keanu.vertices.generictensor.nonprobabilistic;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.generic.SimpleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

public class ConstantVertex<T, TENSOR extends Tensor<T>> extends NonProbabilistic<T, TENSOR> {

    public static ConstantVertex<Boolean, BooleanTensor> of(boolean value) {
        return new ConstantVertex<>(BooleanTensor.scalar(value));
    }

    public static ConstantVertex<Integer, IntegerTensor> of(int value) {
        return new ConstantVertex<>(IntegerTensor.scalar(value));
    }

    public static ConstantVertex<Double, DoubleTensor> of(double value) {
        return new ConstantVertex<>(DoubleTensor.scalar(value));
    }

    public static <GENERIC> ConstantVertex<GENERIC, SimpleTensor<GENERIC>> of(GENERIC value) {
        return new ConstantVertex<>(new SimpleTensor<>(value));
    }

    public ConstantVertex(TENSOR value) {
        setValue(value);
    }

    @Override
    public TENSOR sample(KeanuRandom random) {
        return getValue();
    }

    @Override
    public TENSOR getDerivedValue() {
        return getValue();
    }
}
