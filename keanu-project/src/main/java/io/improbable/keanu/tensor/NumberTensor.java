package io.improbable.keanu.tensor;

import java.util.function.Function;

import io.improbable.keanu.kotlin.Operators;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;

public interface NumberTensor<N extends Number, T extends NumberTensor<N,T>> extends Tensor<N>, Operators<T> {

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

    DoubleTensor getGreaterThanMask(T greaterThanThis);

    DoubleTensor getGreaterThanOrEqualToMask(T greaterThanThis);

    DoubleTensor getLessThanMask(T lessThanThis);

    DoubleTensor getLessThanOrEqualToMask(T lessThanThis);

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
