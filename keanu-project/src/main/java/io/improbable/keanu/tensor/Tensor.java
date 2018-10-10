package io.improbable.keanu.tensor;


import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import static org.apache.commons.math3.util.MathArrays.copyOf;

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

        int[] shape = a.getShape();
        return BooleanTensor.create(equality, copyOf(shape, shape.length));
    }

    static DoubleTensor scalar(Double value) {
       return DoubleTensor.scalar(value);
    }

    static DoubleTensor scalar(double value) {
        return DoubleTensor.scalar(value);
    }

    static IntegerTensor scalar(Integer value) {
        return IntegerTensor.scalar(value);
    }

    static IntegerTensor scalar(int value) {
        return IntegerTensor.scalar(value);
    }

    static BooleanTensor scalar(Boolean value) {
        return BooleanTensor.scalar(value);
    }

    static BooleanTensor scalar(boolean value) {
        return BooleanTensor.scalar(value);
    }

    static <T> Tensor<T> scalar(T value) {
        return GenericTensor.scalar(value);
    }

    static <T> Tensor<T> placeHolder(int[] shape) {
        return new GenericTensor<>(shape);
    }

    static DoubleTensor create(Double value, int[] shape) {
        return DoubleTensor.create(value, shape);
    }

    static DoubleTensor create(double value, int[] shape) {
        return DoubleTensor.create(value, shape);
    }

    static IntegerTensor create(Integer value, int[] shape) {
        return IntegerTensor.create(value, shape);
    }

    static IntegerTensor create(int value, int[] shape) {
        return IntegerTensor.create(value, shape);
    }

    static BooleanTensor create(Boolean value, int[] shape) {
        return BooleanTensor.create(value, shape);
    }

    static BooleanTensor create(boolean value, int[] shape) {
        return BooleanTensor.create(value, shape);
    }

    static <T> GenericTensor<T> create(T value, int[] shape) {
        return GenericTensor.create(value, shape);
    }

    int[] SCALAR_SHAPE = new int[]{1, 1};
    int[] SCALAR_STRIDE = new int[]{1};

    int getRank();

    int[] getShape();

    long getLength();

    boolean isShapePlaceholder();

    T getValue(int... index);

    Tensor<T> setValue(T value, int... index);

    T scalar();

    Tensor<T> duplicate();

    Tensor<T> slice(int dimension, int index);

    double[] asFlatDoubleArray();

    int[] asFlatIntegerArray();

    T[] asFlatArray();

    Tensor<T> reshape(int... newShape);

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

    default boolean hasSameShapeAs(int[] shape) {
        return Arrays.equals(this.getShape(), shape);
    }

    default BooleanTensor elementwiseEquals(Tensor that) {
        return elementwiseEquals(this, that);
    }

}
