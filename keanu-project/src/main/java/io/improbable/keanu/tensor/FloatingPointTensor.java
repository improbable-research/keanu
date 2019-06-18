package io.improbable.keanu.tensor;

import io.improbable.keanu.kotlin.DoubleOperators;
import io.improbable.keanu.tensor.bool.BooleanTensor;

public interface FloatingPointTensor<N extends Number, T extends FloatingPointTensor<N, T>> extends NumberTensor<N, T>, DoubleOperators<T> {

    default T replaceNaN(N value) {
        return duplicate().replaceNaNInPlace(value);
    }

    T replaceNaNInPlace(N value);

    BooleanTensor notNaN();

    default BooleanTensor isNaN() {
        return notNaN().not();
    }

    default T reciprocal() {
        return duplicate().reciprocalInPlace();
    }

    T reciprocalInPlace();

    default T sqrt() {
        return duplicate().sqrtInPlace();
    }

    T sqrtInPlace();

    default T log() {
        return duplicate().logInPlace();
    }

    T logInPlace();

    default T logGamma() {
        return duplicate().logGammaInPlace();
    }

    T logGammaInPlace();

    default T digamma() {
        return duplicate().digammaInPlace();
    }

    T digammaInPlace();

    default T sin() {
        return duplicate().sinInPlace();
    }

    T sinInPlace();

    default T cos() {
        return duplicate().cosInPlace();
    }

    T cosInPlace();

    default T tan() {
        return duplicate().tanInPlace();
    }

    T tanInPlace();

    default T atan() {
        return duplicate().atanInPlace();
    }

    T atanInPlace();

    default T atan2(N y) {
        return duplicate().atan2InPlace(y);
    }

    T atan2InPlace(N y);

    default T atan2(T y) {
        return duplicate().atan2InPlace(y);
    }

    T atan2InPlace(T y);

    default T asin() {
        return duplicate().asinInPlace();
    }

    T asinInPlace();

    default T acos() {
        return duplicate().acosInPlace();
    }

    T acosInPlace();

    default T exp() {
        return duplicate().expInPlace();
    }

    T expInPlace();

    T matrixInverse();

    default T ceil() {
        return duplicate().ceilInPlace();
    }

    T ceilInPlace();

    default T floor() {
        return duplicate().floorInPlace();
    }

    T floorInPlace();

    /**
     * @return The tensor with the elements rounded half up
     * e.g. 1.5 is 2
     * e.g. -2.5 is -3
     */

    default T round() {
        return duplicate().roundInPlace();
    }

    T roundInPlace();

    default T sigmoid() {
        return duplicate().sigmoidInPlace();
    }

    T sigmoidInPlace();

    T choleskyDecomposition();

    N determinant();

    default T standardize() {
        return duplicate().standardizeInPlace();
    }

    T standardizeInPlace();
}
