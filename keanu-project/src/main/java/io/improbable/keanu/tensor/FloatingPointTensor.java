package io.improbable.keanu.tensor;

public interface FloatingPointTensor<N extends Number, T extends FloatingPointTensor<N, T>> extends NumberTensor<N, T> {


    //New tensor Ops and transforms

    T reciprocal();

    T minus(N value);

    T reverseMinus(N value);

    T plus(N value);

    T times(N value);

    T div(N value);

    T powInPlace(N exponent);

    T divInPlace(N value);

    T timesInPlace(N value);

    T plusInPlace(N value);

    T minusInPlace(N value);

    T reverseDiv(N value);

    T matrixMultiply(T value);

    T tensorMultiply(T value, int[] dimsLeft, int[] dimsRight);

    T pow(T exponent);

    T pow(N exponent);

    T sqrt();

    T log();

    T safeLogTimes(T y);

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

    T exp();

    T matrixInverse();

    N average();

    N standardDeviation();

    T standardize();

    T replaceNaN(N value);

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

    N determinant();

}
