package io.improbable.keanu.tensor;

import org.nd4j.linalg.api.ndarray.INDArray;

public abstract class Nd4jFixedPointTensor<T extends Number, TENSOR extends FixedPointTensor<T, TENSOR>> extends Nd4jNumberTensor<T, TENSOR> implements FixedPointTensor<T, TENSOR> {

    public Nd4jFixedPointTensor(INDArray tensor) {
        super(tensor);
    }
}
