package io.improbable.keanu.tensor;


import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.generic.GenericTensor;

import java.util.Arrays;
import java.util.List;

import static org.apache.commons.math3.util.MathArrays.copyOf;

public interface Tensor<T> {

    int[] SCALAR_SHAPE = new int[]{1, 1};
    int[] SCALAR_STRIDE = new int[]{1};

    int getRank();

    int[] getShape();

    long getLength();

    boolean isShapePlaceholder();

    default boolean isScalar() {
        return getLength() == 1;
    }

    default boolean isVector() {
        return getRank() == 1;
    }

    default boolean isMatrix() {
        return getRank() == 2;
    }

    default boolean hasSameShapeAs(Tensor that) {
        return hasSameShapeAs(that.getShape());
    }

    default boolean hasSameShapeAs(int[] shape) {
        return Arrays.equals(this.getShape(), shape);
    }

    T getValue(int... index);

    void setValue(T value, int... index);

    T scalar();

    Tensor<T> duplicate();

    FlattenedView<T> getFlattenedView();

    interface FlattenedView<T> {

        long size();

        T get(long index);

        T getOrScalar(long index);

        void set(long index, T value);
    }

    double[] asFlatDoubleArray();

    int[] asFlatIntegerArray();

    T[] asFlatArray();

    default List<T> asFlatList() {
        return Arrays.asList(asFlatArray());
    }

    default BooleanTensor elementwiseEquals(Tensor that) {
        return elementwiseEquals(this, that);
    }

    static BooleanTensor elementwiseEquals(Tensor a, Tensor b) {
        if (!a.hasSameShapeAs(b)) {
            throw new IllegalArgumentException("Cannot compare tensors of different shapes");
        }

        Object[] aArray = a.asFlatArray();
        Object[] bArray = b.asFlatArray();

        boolean[] equality = new boolean[aArray.length];

        for (int i = 0; i < aArray.length; i++) {
            equality[i] = aArray[i].equals(bArray[i]);
        }

        int[] shape = a.getShape();
        return BooleanTensor.create(equality, copyOf(shape, shape.length));
    }

    static <T> Tensor<T> scalar(T value) {
        return new GenericTensor<>(value);
    }

    static <T> Tensor<T> placeHolder(int[] shape) {
        return new GenericTensor<>(shape);
    }
}
