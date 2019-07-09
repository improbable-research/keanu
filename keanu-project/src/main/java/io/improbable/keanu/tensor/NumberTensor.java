package io.improbable.keanu.tensor;

import io.improbable.keanu.BaseNumberTensor;
import io.improbable.keanu.kotlin.NumberOperators;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;

import java.util.function.Function;

public interface NumberTensor<N extends Number, T extends NumberTensor<N, T>>
    extends Tensor<N, T>, BaseNumberTensor<BooleanTensor, IntegerTensor, DoubleTensor, N, T>, NumberOperators<T> {

    @Override
    BooleanTensor toBoolean();

    @Override
    DoubleTensor toDouble();

    @Override
    IntegerTensor toInteger();

    double[] asFlatDoubleArray();

    int[] asFlatIntegerArray();

    @Override
    default T cumSum(int requestedDimension) {
        return duplicate().cumSumInPlace(requestedDimension);
    }

    T cumSumInPlace(int dimension);

    @Override
    default T cumProd(int requestedDimension) {
        return duplicate().cumProdInPlace(requestedDimension);
    }

    T cumProdInPlace(int dimension);

    @Override
    default T max(T max) {
        return duplicate().maxInPlace(max);
    }

    T maxInPlace(T max);

    @Override
    default T min(T min) {
        return duplicate().minInPlace(min);
    }

    T minInPlace(T min);

    @Override
    default T clamp(T min, T max) {
        return duplicate().clampInPlace(min, max);
    }

    T clampInPlace(T min, T max);

    @Override
    default T abs() {
        return duplicate().absInPlace();
    }

    T absInPlace();

    @Override
    default T minus(N value) {
        return duplicate().minusInPlace(value);
    }

    T minusInPlace(N value);

    @Override
    default T minus(T that) {
        return duplicate().minusInPlace(that);
    }

    T minusInPlace(T that);

    @Override
    default T reverseMinus(T value) {
        return duplicate().reverseMinusInPlace(value);
    }

    T reverseMinusInPlace(T value);

    @Override
    default T reverseMinus(N value) {
        return duplicate().reverseMinusInPlace(value);
    }

    T reverseMinusInPlace(N value);

    @Override
    default T plus(N value) {
        return duplicate().plusInPlace(value);
    }

    T plusInPlace(N value);

    @Override
    default T plus(T that) {
        return duplicate().plusInPlace(that);
    }

    T plusInPlace(T that);

    @Override
    default T unaryMinus() {
        return duplicate().unaryMinusInPlace();
    }

    T unaryMinusInPlace();

    @Override
    default T times(N value) {
        return duplicate().timesInPlace(value);
    }

    T timesInPlace(N value);

    @Override
    default T times(T that) {
        return duplicate().timesInPlace(that);
    }

    T timesInPlace(T that);

    @Override
    default T div(N value) {
        return duplicate().divInPlace(value);
    }

    T divInPlace(N value);

    @Override
    default T div(T value) {
        return duplicate().divInPlace(value);
    }

    T divInPlace(T that);

    @Override
    default T reverseDiv(N value) {
        return duplicate().reverseDivInPlace(value);
    }

    T reverseDivInPlace(N value);

    @Override
    default T reverseDiv(T value) {
        return duplicate().reverseDivInPlace(value);
    }

    T reverseDivInPlace(T value);

    @Override
    default T pow(T exponent) {
        return duplicate().powInPlace(exponent);
    }

    T powInPlace(T exponent);

    @Override
    default T pow(N exponent) {
        return duplicate().powInPlace(exponent);
    }

    T powInPlace(N exponent);

    @Override
    default T setWithMask(T mask, N value) {
        return duplicate().setWithMaskInPlace(mask, value);
    }

    T setWithMaskInPlace(T mask, N value);

    @Override
    default T apply(Function<N, N> function) {
        return duplicate().applyInPlace(function);
    }

    T applyInPlace(Function<N, N> function);

    T setAllInPlace(N value);

    /**
     * This is identical to log().times(y), except that it changes NaN results to 0.
     * This is important when calculating 0log0, which should return 0
     * See https://arcsecond.wordpress.com/2009/03/19/0log0-0-for-real/ for some mathematical justification
     *
     * @param y The tensor value to multiply by
     * @return the log of this tensor multiplied by y
     */
    @Override
    default T safeLogTimes(T y) {
        return duplicate().safeLogTimesInPlace(y);
    }

    T safeLogTimesInPlace(T y);

    @Override
    BooleanTensor equalsWithinEpsilon(T other, N epsilon);

    @Override
    BooleanTensor lessThan(T value);

    @Override
    BooleanTensor lessThanOrEqual(T value);

    @Override
    BooleanTensor greaterThan(T value);

    @Override
    BooleanTensor greaterThanOrEqual(T value);

    @Override
    BooleanTensor lessThan(N value);

    @Override
    BooleanTensor lessThanOrEqual(N value);

    @Override
    BooleanTensor greaterThan(N value);

    @Override
    BooleanTensor greaterThanOrEqual(N value);

    N sumNumber();

}
