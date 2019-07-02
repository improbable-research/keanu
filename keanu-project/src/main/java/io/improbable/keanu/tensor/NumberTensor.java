package io.improbable.keanu.tensor;

import io.improbable.keanu.kotlin.NumberOperators;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;

import java.util.function.Function;

public interface NumberTensor<N extends Number, T extends NumberTensor<N, T>> extends Tensor<N, T>, NumberOperators<T> {

    DoubleTensor toDouble();

    IntegerTensor toInteger();

    double[] asFlatDoubleArray();

    int[] asFlatIntegerArray();

    N sum();

    T sum(int... overDimensions);

    default T cumSum(int requestedDimension) {
        return duplicate().cumSumInPlace(requestedDimension);
    }

    T cumSumInPlace(int dimension);

    N product();

    T product(int... overDimensions);

    default T cumProd(int requestedDimension) {
        return duplicate().cumProdInPlace(requestedDimension);
    }

    T cumProdInPlace(int dimension);

    N max();

    default T max(T max) {
        return duplicate().maxInPlace(max);
    }

    T maxInPlace(T max);

    N min();

    default T min(T min) {
        return duplicate().minInPlace(min);
    }

    T minInPlace(T min);

    default T clamp(T min, T max) {
        return duplicate().clampInPlace(min, max);
    }

    T clampInPlace(T min, T max);

    T matrixMultiply(T value);

    T tensorMultiply(T value, int[] dimLeft, int[] dimsRight);

    default T abs() {
        return duplicate().absInPlace();
    }

    T absInPlace();

    default T minus(N value) {
        return duplicate().minusInPlace(value);
    }

    T minusInPlace(N value);

    default T minus(T that) {
        return duplicate().minusInPlace(that);
    }

    T minusInPlace(T that);

    default T reverseMinus(T value) {
        return duplicate().reverseMinusInPlace(value);
    }

    T reverseMinusInPlace(T value);

    default T reverseMinus(N value) {
        return duplicate().reverseMinusInPlace(value);
    }

    T reverseMinusInPlace(N value);

    default T plus(N value) {
        return duplicate().plusInPlace(value);
    }

    T plusInPlace(N value);

    default T plus(T that) {
        return duplicate().plusInPlace(that);
    }

    T plusInPlace(T that);

    default T unaryMinus() {
        return duplicate().unaryMinusInPlace();
    }

    T unaryMinusInPlace();

    default T times(N value) {
        return duplicate().timesInPlace(value);
    }

    T timesInPlace(N value);

    default T times(T that) {
        return duplicate().timesInPlace(that);
    }

    T timesInPlace(T that);

    default T div(N value) {
        return duplicate().divInPlace(value);
    }

    T divInPlace(N value);

    default T div(T value) {
        return duplicate().divInPlace(value);
    }

    T divInPlace(T that);

    default T reverseDiv(N value) {
        return duplicate().reverseDivInPlace(value);
    }

    T reverseDivInPlace(N value);

    default T reverseDiv(T value) {
        return duplicate().reverseDivInPlace(value);
    }

    T reverseDivInPlace(T value);

    default T pow(T exponent) {
        return duplicate().powInPlace(exponent);
    }

    T powInPlace(T exponent);

    default T pow(N exponent) {
        return duplicate().powInPlace(exponent);
    }

    T powInPlace(N exponent);

    N average();

    N standardDeviation();

    /**
     * Find the index into the flattened array of the tensor of the largest value, e.g.
     * <pre>
     * DoubleTensor tensor = DoubleTensor.arange(0, 6).reshape(2, 3);
     * // [[0., 1., 2.],
     * //  [3., 4., 5.]]
     * IntegerTensor max = tensor.argMax();
     * // [[5]]
     * </pre>
     *
     * @return A scalar tensor with the index of the largest value. If there are multiple largest values, it will be
     * the first index.
     */
    int argMax();

    /**
     * Find the indices into the tensor of the largest values in a specified axis (dimension), e.g.
     * <pre>
     * DoubleTensor tensor = DoubleTensor.arange(0, 6).reshape(2, 3);
     * // [[0., 1., 2.],
     * //  [3., 4., 5.]]
     * IntegerTensor maxesFor0 = tensor.argMax(0);
     * // [[1, 1, 1]]
     * IntegerTensor maxFor1 = tensor.argMax(1);
     * // [[2, 2]]
     * </pre>
     *
     * @param axis The axis (dimension) to find the largest values in
     * @return A tensor where each value is the location of the maximum value in the vector at that location in the
     * specified dimension in the original tensor. If there are multiple largest values in this vector, it will
     * be the first index.
     * @see <a href="https://www.geeksforgeeks.org/numpy-argmax-python/">An article about argmax over an axis in numpy</a>
     */
    IntegerTensor argMax(int axis);

    int argMin();

    IntegerTensor argMin(int axis);

    default T setWithMask(T mask, N value) {
        return duplicate().setWithMaskInPlace(mask, value);
    }

    T setWithMaskInPlace(T mask, N value);

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
    default T safeLogTimes(T y) {
        return duplicate().safeLogTimesInPlace(y);
    }

    T safeLogTimesInPlace(T y);

    // Comparisons

    boolean equalsWithinEpsilon(T other, N epsilon);

    BooleanTensor lessThan(T value);

    BooleanTensor lessThanOrEqual(T value);

    BooleanTensor greaterThan(T value);

    BooleanTensor greaterThanOrEqual(T value);

    BooleanTensor lessThan(N value);

    BooleanTensor lessThanOrEqual(N value);

    BooleanTensor greaterThan(N value);

    BooleanTensor greaterThanOrEqual(N value);

    T greaterThanMask(T greaterThanThis);

    T greaterThanOrEqualToMask(T greaterThanThis);

    T lessThanMask(T lessThanThis);

    T lessThanOrEqualToMask(T lessThanThis);
}
