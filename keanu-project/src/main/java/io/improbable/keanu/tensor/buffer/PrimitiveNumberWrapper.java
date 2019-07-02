package io.improbable.keanu.tensor.buffer;

public interface PrimitiveNumberWrapper<T extends Number, B extends JVMBuffer.PrimitiveArrayWrapper<T, B>> extends JVMBuffer.PrimitiveArrayWrapper<T, B> {

    int[] asIntegerArray();

    double[] asDoubleArray();

    T sum();

    T product();

    B times(T that);

    B times(int index, T that);

    B div(T that);

    B plus(T that);

    B plus(int index, T that);

    B minus(T that);

    B pow(T that);

    B reverseDiv(T that);

    B reverseMinus(T that);
}
