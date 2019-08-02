package io.improbable.keanu.tensor;

public interface FixedPointTensor<N extends Number, T extends FixedPointTensor<N, T>> extends NumberTensor<N, T> {

    default T mod(N that) {
        return duplicate().modInPlace(that);
    }

    T modInPlace(N that);

    default T mod(T that) {
        return duplicate().modInPlace(that);
    }

    T modInPlace(T that);
}
