package io.improbable.keanu.tensor;

import org.nd4j.linalg.api.ndarray.INDArray;

public abstract class Nd4jFloatingPointTensor<T extends Number, TENSOR extends FloatingPointTensor<T, TENSOR>> extends Nd4jNumberTensor<T, TENSOR> implements FloatingPointTensor<T, TENSOR> {

    public Nd4jFloatingPointTensor(INDArray tensor) {
        super(tensor);
    }
}
