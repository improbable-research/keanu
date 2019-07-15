package io.improbable.keanu;

public interface BaseFloatingPointTensor<
    BOOLEAN extends BaseTensor<BOOLEAN, Boolean, BOOLEAN>,
    INTEGER extends BaseNumberTensor<BOOLEAN, INTEGER, DOUBLE, Integer, INTEGER>,
    DOUBLE extends BaseNumberTensor<BOOLEAN, INTEGER, DOUBLE, Double, DOUBLE>,
    N extends Number,
    T extends BaseFloatingPointTensor<BOOLEAN, INTEGER, DOUBLE, N, T>
    > extends BaseNumberTensor<BOOLEAN, INTEGER, DOUBLE, N, T> {

    T replaceNaN(N value);

    BOOLEAN notNaN();

    BOOLEAN isNaN();

    BOOLEAN isFinite();

    BOOLEAN isInfinite();

    BOOLEAN isNegativeInfinity();

    BOOLEAN isPositiveInfinity();

    INTEGER nanArgMax(int axis);

    INTEGER nanArgMax();

    INTEGER nanArgMin(int axis);

    INTEGER nanArgMin();

    T reciprocal();

    T sqrt();

    T log();

    T logGamma();

    T digamma();

    T sin();

    T cos();

    T tan();

    T atan();

    T atan2(N y);

    T atan2(T y);

    T asin();

    T acos();

    T sinh();

    T cosh();

    T tanh();

    T asinh();

    T acosh();

    T atanh();

    T exp();

    T logAddExp2(T that);

    T logAddExp(T that);

    T log1p();

    T log2();

    T log10();

    T exp2();

    T expM1();

    T ceil();

    T floor();

    /**
     * @return The tensor with the elements rounded half up
     * e.g. 1.5 is 2
     * e.g. -2.5 is -3
     */
    T round();

    T sigmoid();

    T choleskyDecomposition();

    T determinant();

    T matrixInverse();

    T standardize();

    T average();

    T standardDeviation();
}
