package io.improbable.keanu.vertices.generictensor.nonprobabilistic;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.generic.SimpleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

public class ConstantVertex<TENSOR extends Tensor> extends NonProbabilistic<TENSOR> {

    public static ConstantVertex<BooleanTensor> of(boolean value) {
        return new ConstantVertex<>(BooleanTensor.scalar(value));
    }

    public static ConstantVertex<IntegerTensor> of(int value) {
        return new ConstantVertex<>(IntegerTensor.scalar(value));
    }

    public static ConstantVertex<DoubleTensor> of(double value) {
        return new ConstantVertex<>(DoubleTensor.scalar(value));
    }

    public static <GENERIC> ConstantVertex<SimpleTensor<GENERIC>> of(GENERIC value) {
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
