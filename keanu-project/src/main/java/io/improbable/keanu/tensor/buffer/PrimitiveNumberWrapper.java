package io.improbable.keanu.tensor.buffer;

public interface PrimitiveNumberWrapper<T extends Number, B extends JVMBuffer.PrimitiveArrayWrapper<T, B>> extends JVMBuffer.PrimitiveArrayWrapper<T, B> {

    int[] asIntegerArray();

    double[] asDoubleArray();

    T sum();

    void times(T that);

    void div(T that);

    void plus(T that);

    void plus(int index, T that);

    void minus(T that);

    void pow(T that);

    void reverseDiv(T that);

    void reverseMinus(T that);
}
