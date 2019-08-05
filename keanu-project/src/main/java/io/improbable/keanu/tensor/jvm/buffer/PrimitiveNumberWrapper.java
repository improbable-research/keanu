package io.improbable.keanu.tensor.jvm.buffer;

import io.improbable.keanu.tensor.bool.BooleanBuffer;

public interface PrimitiveNumberWrapper<T extends Number, B extends JVMBuffer.PrimitiveArrayWrapper<T, B>> extends JVMBuffer.PrimitiveArrayWrapper<T, B> {

    int[] asIntegerArray();

    long[] asLongArray();

    double[] asDoubleArray();

    T sum();

    T product();

    B times(T that);

    B times(long index, T that);

    B div(T that);

    B plus(T that);

    B plus(long index, T that);

    B minus(T that);

    B pow(T that);

    B reverseDiv(T that);

    B reverseMinus(T that);

    BooleanBuffer.PrimitiveBooleanWrapper greaterThan(T that);

    BooleanBuffer.PrimitiveBooleanWrapper lessThan(T that);

    BooleanBuffer.PrimitiveBooleanWrapper greaterThanOrEqual(T that);

    BooleanBuffer.PrimitiveBooleanWrapper lessThanOrEqual(T that);
}
