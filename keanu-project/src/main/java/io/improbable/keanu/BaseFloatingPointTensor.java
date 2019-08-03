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

    /**
     * This is identical to log().times(y), except that it changes NaN results to 0.
     * This is important when calculating 0log0, which should return 0
     * See https://arcsecond.wordpress.com/2009/03/19/0log0-0-for-real/ for some mathematical justification
     *
     * @param y The tensor value to multiply by
     * @return the log of this tensor multiplied by y
     */
    T safeLogTimes(T y);

    T logGamma();

    T digamma();

    T trigamma();

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

    T matrixDeterminant();

    T matrixInverse();

    T standardize();

    /**
     * Mean over all dimensions. This will always result in a scalar.
     *
     * @return the mean of all elements
     */
    T mean();

    /**
     * Mean over specified dimensions.
     *
     * @param overDimensions dimensions to mean over.
     * @return the mean result
     */
    T mean(int... overDimensions);

    T standardDeviation();
}
