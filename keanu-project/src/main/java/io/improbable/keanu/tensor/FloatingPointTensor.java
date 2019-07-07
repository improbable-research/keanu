package io.improbable.keanu.tensor;

import io.improbable.keanu.BaseFloatingPointTensor;
import io.improbable.keanu.kotlin.DoubleOperators;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;

public interface FloatingPointTensor<N extends Number, T extends FloatingPointTensor<N, T>>
    extends NumberTensor<N, T>, BaseFloatingPointTensor<BooleanTensor, IntegerTensor, DoubleTensor, N, T>, DoubleOperators<T> {

    @Override
    default T replaceNaN(N value) {
        return duplicate().replaceNaNInPlace(value);
    }

    T replaceNaNInPlace(N value);

    @Override
    default T reciprocal() {
        return duplicate().reciprocalInPlace();
    }

    T reciprocalInPlace();

    @Override
    default T sqrt() {
        return duplicate().sqrtInPlace();
    }

    T sqrtInPlace();

    @Override
    default T log() {
        return duplicate().logInPlace();
    }

    T logInPlace();

    @Override
    default T logGamma() {
        return duplicate().logGammaInPlace();
    }

    T logGammaInPlace();

    @Override
    default T digamma() {
        return duplicate().digammaInPlace();
    }

    T digammaInPlace();

    @Override
    default T sin() {
        return duplicate().sinInPlace();
    }

    T sinInPlace();

    @Override
    default T cos() {
        return duplicate().cosInPlace();
    }

    T cosInPlace();

    @Override
    default T tan() {
        return duplicate().tanInPlace();
    }

    T tanInPlace();

    @Override
    default T atan() {
        return duplicate().atanInPlace();
    }

    T atanInPlace();

    @Override
    default T atan2(N y) {
        return duplicate().atan2InPlace(y);
    }

    T atan2InPlace(N y);

    @Override
    default T atan2(T y) {
        return duplicate().atan2InPlace(y);
    }

    T atan2InPlace(T y);

    @Override
    default T asin() {
        return duplicate().asinInPlace();
    }

    T asinInPlace();

    @Override
    default T acos() {
        return duplicate().acosInPlace();
    }

    T acosInPlace();

    @Override
    default T sinh() {
        return duplicate().sinhInPlace();
    }

    T sinhInPlace();

    @Override
    default T cosh() {
        return duplicate().coshInPlace();
    }

    T coshInPlace();

    @Override
    default T tanh() {
        return duplicate().tanhInPlace();
    }

    T tanhInPlace();

    @Override
    default T asinh() {
        return duplicate().asinhInPlace();
    }

    T asinhInPlace();

    @Override
    default T acosh() {
        return duplicate().acoshInPlace();
    }

    T acoshInPlace();

    @Override
    default T atanh() {
        return duplicate().atanhInPlace();
    }

    T atanhInPlace();

    @Override
    default T exp() {
        return duplicate().expInPlace();
    }

    T expInPlace();

    @Override
    default T logAddExp2(T that) {
        return duplicate().logAddExp2InPlace(that);
    }

    T logAddExp2InPlace(T that);

    @Override
    default T logAddExp(T that) {
        return duplicate().logAddExpInPlace(that);
    }

    T logAddExpInPlace(T that);

    @Override
    default T log1p() {
        return duplicate().log1pInPlace();
    }

    T log1pInPlace();

    @Override
    default T log2() {
        return duplicate().log2InPlace();
    }

    T log2InPlace();

    @Override
    default T log10() {
        return duplicate().log10InPlace();
    }

    T log10InPlace();

    @Override
    default T exp2() {
        return duplicate().exp2InPlace();
    }

    T exp2InPlace();

    @Override
    default T expM1() {
        return duplicate().expM1InPlace();
    }

    T expM1InPlace();

    @Override
    default T ceil() {
        return duplicate().ceilInPlace();
    }

    T ceilInPlace();

    @Override
    default T floor() {
        return duplicate().floorInPlace();
    }

    T floorInPlace();

    /**
     * @return The tensor with the elements rounded half up
     * e.g. 1.5 is 2
     * e.g. -2.5 is -3
     */
    @Override
    default T round() {
        return duplicate().roundInPlace();
    }

    T roundInPlace();

    @Override
    default T sigmoid() {
        return duplicate().sigmoidInPlace();
    }

    T sigmoidInPlace();

    @Override
    default T standardize() {
        return duplicate().standardizeInPlace();
    }

    T standardizeInPlace();
}
