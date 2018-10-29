package io.improbable.keanu.tensor;

import io.improbable.keanu.kotlin.NumberOperators;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;

import java.util.function.Function;

public interface NumberTensor<N extends Number, T extends NumberTensor<N,T>> extends Tensor<N>, NumberOperators<T> {

    N sum();

    DoubleTensor toDouble();

    IntegerTensor toInteger();

    T diag();

    T transpose();

    T sum(int... overDimensions);
    
    // New tensor Ops and transforms
    
    T matrixMultiply(T value);

    T tensorMultiply(T value, int[] dimLeft, int[] dimsRight);
    
    T abs();

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
     *         the first index.
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
     *         specified dimension in the original tensor. If there are multiple largest values in this vector, it will
     *         be the first index.
     * @see <a href="https://www.geeksforgeeks.org/numpy-argmax-python/">An article about argmax over an axis in numpy</a>
     */
    IntegerTensor argMax(int axis);

    T getGreaterThanMask(T greaterThanThis);

    T getGreaterThanOrEqualToMask(T greaterThanThis);

    T getLessThanMask(T lessThanThis);

    T getLessThanOrEqualToMask(T lessThanThis);

    T setWithMaskInPlace(T mask, N value);

    T setWithMask(T mask, N value);

    T apply(Function<N, N> function);
    
    // In Place

    T minusInPlace(T that);

    T plusInPlace(T that);

    T timesInPlace(T that);

    T divInPlace(T that);

    T powInPlace(T exponent);

    T unaryMinusInPlace();

    T absInPlace();

    T applyInPlace(Function<N, N> function);

    // Comparisons

    BooleanTensor lessThan(T value);

    BooleanTensor lessThanOrEqual(T value);

    BooleanTensor greaterThan(T value);

    BooleanTensor greaterThanOrEqual(T value);
}
