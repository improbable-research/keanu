package io.improbable.keanu.tensor;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;

public interface NumberTensor<T extends Number> extends Tensor<T> {

    T sum();

    DoubleTensor toDouble();

    IntegerTensor toInteger();

}
