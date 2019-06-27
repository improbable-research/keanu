package io.improbable.keanu.tensor;

import io.improbable.keanu.kotlin.DoubleOperators;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;

public interface FloatingPointTensor<N extends Number, T extends FloatingPointTensor<N, T>> extends NumberTensor<N, T>, DoubleOperators<T> {

    default T replaceNaN(N value) {
        return duplicate().replaceNaNInPlace(value);
    }

    T replaceNaNInPlace(N value);

    BooleanTensor notNaN();

    default BooleanTensor isNaN() {
        return notNaN().not();
    }

    IntegerTensor nanArgMax(int axis);

    int nanArgMax();

    IntegerTensor nanArgMin(int axis);

    int nanArgMin();

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

    default T sinh() {
        return duplicate().sinhInPlace();
    }

    T sinhInPlace();

    default T cosh() {
        return duplicate().coshInPlace();
    }

    T coshInPlace();

    default T tanh() {
        return duplicate().tanhInPlace();
    }

    T tanhInPlace();

    default T asinh() {
        return duplicate().asinhInPlace();
    }

    T asinhInPlace();

    default T acosh() {
        return duplicate().acoshInPlace();
    }

    T acoshInPlace();

    default T atanh() {
        return duplicate().atanhInPlace();
    }

    T atanhInPlace();

    default T exp() {
        return duplicate().expInPlace();
    }

    T expInPlace();

    default T logAddExp2(T that) {
        return duplicate().logAddExp2InPlace(that);
    }

    T logAddExp2InPlace(T that);

    default T logAddExp(T that) {
        return duplicate().logAddExpInPlace(that);
    }

    T logAddExpInPlace(T that);

    default T log1p() {
        return duplicate().log1pInPlace();
    }

    T log1pInPlace();

    default T log2() {
        return duplicate().log2InPlace();
    }

    T log2InPlace();

    default T log10() {
        return duplicate().log10InPlace();
    }

    T log10InPlace();

    default T exp2() {
        return duplicate().exp2InPlace();
    }

    T exp2InPlace();

    default T expM1() {
        return duplicate().expM1InPlace();
    }

    T expM1InPlace();

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

    T matrixInverse();

    default T standardize() {
        return duplicate().standardizeInPlace();
    }

    T standardizeInPlace();
}
