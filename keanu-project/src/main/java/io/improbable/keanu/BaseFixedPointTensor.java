package io.improbable.keanu;

public interface BaseFixedPointTensor<
    BOOLEAN extends BaseTensor<BOOLEAN, Boolean, BOOLEAN>,
    INTEGER extends BaseNumberTensor<BOOLEAN, INTEGER, DOUBLE, Integer, INTEGER>,
    DOUBLE extends BaseNumberTensor<BOOLEAN, INTEGER, DOUBLE, Double, DOUBLE>,
    N extends Number,
    T extends BaseFixedPointTensor<BOOLEAN, INTEGER, DOUBLE, N, T>
    > extends BaseNumberTensor<BOOLEAN, INTEGER, DOUBLE, N, T> {

    T mod(N that);

    T mod(T that);
}
