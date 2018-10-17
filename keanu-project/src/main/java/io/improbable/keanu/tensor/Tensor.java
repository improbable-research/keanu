package io.improbable.keanu.tensor;


import java.util.Arrays;
import java.util.List;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.generic.GenericTensor;

public interface Tensor<T> {

    static BooleanTensor elementwiseEquals(Tensor a, Tensor b) {
        if (!a.hasSameShapeAs(b)) {
            throw new IllegalArgumentException(
                String.format("Cannot compare tensors of different shapes %s and %s",
                    Arrays.toString(a.getShape()), Arrays.toString(b.getShape()))
            );
        }

        Object[] aArray = a.asFlatArray();
        Object[] bArray = b.asFlatArray();

        boolean[] equality = new boolean[aArray.length];

        for (int i = 0; i < aArray.length; i++) {
            equality[i] = aArray[i].equals(bArray[i]);
        }

        long[] shape = a.getShape();
        return BooleanTensor.create(equality, Arrays.copyOf(shape, shape.length));
    }

    static <T> Tensor<T> scalar(T value) {
        return GenericTensor.scalar(value);
    }

    static <T> Tensor<T> placeHolder(long[] shape) {
        return new GenericTensor<>(shape);
    }

    static <T> GenericTensor<T> create(T value, long[] shape) {
        return GenericTensor.create(value, shape);
    }

    long[] SCALAR_SHAPE = new long[]{1, 1};
    long[] SCALAR_STRIDE = new long[]{1};

    int getRank();

    long[] getShape();

    long getLength();

    boolean isShapePlaceholder();

    T getValue(long... index);

    default T getValue(int... index) {
        return getValue(Arrays.stream(index).mapToLong(i -> i).toArray());
    }

    Tensor<T> setValue(T value, long... index);

    T scalar();

    Tensor<T> duplicate();

    Tensor<T> slice(int dimension, long index);

    double[] asFlatDoubleArray();

    int[] asFlatIntegerArray();

    T[] asFlatArray();

    Tensor<T> reshape(long... newShape);

    FlattenedView<T> getFlattenedView();

    interface FlattenedView<T> {

        long size();

        T get(long index);

        T getOrScalar(long index);

        void set(long index, T value);
    }

    default List<T> asFlatList() {
        return Arrays.asList(asFlatArray());
    }

    default boolean isScalar() {
        return getLength() == 1;
    }

    /**
     * Returns true if the tensor is a vector. A vector being a 1xn or a nx1 tensor.
     * <p>
     * (1, 2, 3) is a 1x3 vector.
     * <p>
     * (1)
     * (2)
     * (3) is a 3x1 vector.
     *
     * @return true if the tensor is a vector
     */
    default boolean isVector() {
        return getRank() == 1 || (getRank() == 2 && (getShape()[0] == 1 || getShape()[1] == 1));
    }

    default boolean isMatrix() {
        return getRank() == 2;
    }

    default boolean hasSameShapeAs(Tensor that) {
        return hasSameShapeAs(that.getShape());
    }

    default boolean hasSameShapeAs(long[] shape) {
        return Arrays.equals(this.getShape(), shape);
    }

    default BooleanTensor elementwiseEquals(Tensor that) {
        return elementwiseEquals(this, that);
    }

    BooleanTensor elementwiseEquals(T value);

}
