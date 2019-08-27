package io.improbable.keanu.tensor;

import io.improbable.keanu.BaseFixedPointTensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;

public interface FixedPointTensor<N extends Number, T extends FixedPointTensor<N, T>>
    extends NumberTensor<N, T>, BaseFixedPointTensor<BooleanTensor, IntegerTensor, DoubleTensor, N, T> {

    @Override
    default T mod(N that) {
        return duplicate().modInPlace(that);
    }

    T modInPlace(N that);

    @Override
    default T mod(T that) {
        return duplicate().modInPlace(that);
    }

    T modInPlace(T that);
}
