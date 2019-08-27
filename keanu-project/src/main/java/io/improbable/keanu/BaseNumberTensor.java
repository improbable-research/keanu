package io.improbable.keanu;

import java.util.function.Function;

public interface BaseNumberTensor<
    BOOLEAN extends BaseTensor<BOOLEAN, Boolean, BOOLEAN>,
    INTEGER extends BaseNumberTensor<BOOLEAN, INTEGER, DOUBLE, Integer, INTEGER>,
    DOUBLE extends BaseNumberTensor<BOOLEAN, INTEGER, DOUBLE, Double, DOUBLE>,
    N extends Number,
    T extends BaseNumberTensor<BOOLEAN, INTEGER, DOUBLE, N, T>
    > extends BaseTensor<BOOLEAN, N, T> {

    BOOLEAN toBoolean();

    DOUBLE toDouble();

    INTEGER toInteger();

    /**
     * Sum over all dimensions. This will always result in a scalar.
     *
     * @return the summation result
     */
    T sum();

    /**
     * Sum over specified dimensions.
     *
     * @param overDimensions dimensions to sum over.
     * @return the summation result
     */
    T sum(int... overDimensions);

    T cumSum(int requestedDimension);

    T product();

    T product(int... overDimensions);

    T cumProd(int requestedDimension);

    T max();

    T max(T max);

    T min();

    T min(T min);

    T clamp(T min, T max);

    T matrixMultiply(T that);

    T tensorMultiply(T value, int[] dimLeft, int[] dimsRight);

    T abs();

    T sign();

    T minus(N value);

    T minus(T that);

    T reverseMinus(T value);

    T reverseMinus(N value);

    T plus(N value);

    T plus(T that);

    T unaryMinus();

    T times(N value);

    T times(T that);

    T div(N value);

    T div(T value);

    T reverseDiv(N value);

    T reverseDiv(T value);

    T pow(T exponent);

    T pow(N exponent);

    /**
     * Find the indices into the tensor of the largest values in a specified axis (dimension), e.g.
     * <pre>
     * DoubleTensor tensor = DoubleTensor.arange(0, 6).reshape(2, 3);
     *  [[0., 1., 2.],
     *   [3., 4., 5.]]
     * IntegerTensor maxesFor0 = tensor.argMax(0);
     *  [1, 1, 1]
     * IntegerTensor maxFor1 = tensor.argMax(1);
     *  [2, 2]
     * </pre>
     *
     * @param axis The axis (dimension) to find the largest values in
     * @return A tensor where each value is the location of the maximum value in the vector at that location in the
     * specified dimension in the original tensor. If there are multiple largest values in this vector, it will
     * be the first index.
     * @see <a href="https://www.geeksforgeeks.org/numpy-argmax-python/">An article about argmax over an axis in numpy</a>
     */
    INTEGER argMax(int axis);

    INTEGER argMax();

    INTEGER argMin(int axis);

    INTEGER argMin();

    T setWithMask(T mask, N value);

    T apply(Function<N, N> function);

    BOOLEAN equalsWithinEpsilon(T other, N epsilon);

    BOOLEAN lessThan(T value);

    BOOLEAN lessThanOrEqual(T value);

    BOOLEAN greaterThan(T value);

    BOOLEAN greaterThanOrEqual(T value);

    BOOLEAN lessThan(N value);

    BOOLEAN lessThanOrEqual(N value);

    BOOLEAN greaterThan(N value);

    BOOLEAN greaterThanOrEqual(N value);

    T greaterThanMask(T greaterThanThis);

    T greaterThanOrEqualToMask(T greaterThanOrEqualThis);

    T lessThanMask(T lessThanThis);

    T lessThanOrEqualToMask(T lessThanOrEqualThis);

}
