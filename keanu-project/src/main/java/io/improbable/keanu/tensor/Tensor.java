package io.improbable.keanu.tensor;


import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.generic.GenericTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import org.apache.commons.lang3.ArrayUtils;

import java.util.Arrays;
import java.util.List;

public interface Tensor<T> {

    static <DATA, TENSOR extends Tensor<DATA>> TENSOR scalar(DATA data) {
        if (data instanceof Double) {
            return (TENSOR) DoubleTensor.scalar(((Double) data).doubleValue());
        } else if (data instanceof Integer) {
            return (TENSOR) IntegerTensor.scalar(((Integer) data).intValue());
        } else if (data instanceof Boolean) {
            return (TENSOR) BooleanTensor.scalar(((Boolean) data).booleanValue());
        } else {
            return (TENSOR) GenericTensor.scalar(data);
        }
    }

    static <DATA, TENSOR extends Tensor<DATA>> TENSOR createFilled(DATA data, long[] shape) {
        if (data instanceof Double) {
            return (TENSOR) DoubleTensor.create(((Double) data).doubleValue(), shape);
        } else if (data instanceof Integer) {
            return (TENSOR) IntegerTensor.create(((Integer) data).intValue(), shape);
        } else if (data instanceof Boolean) {
            return (TENSOR) BooleanTensor.create(((Boolean) data).booleanValue(), shape);
        } else {
            return (TENSOR) GenericTensor.createFilled(data, shape);
        }
    }

    static <DATA, TENSOR extends Tensor<DATA>> TENSOR create(DATA[] data, long[] shape) {
        if (data instanceof Double[]) {
            return (TENSOR) DoubleTensor.create(ArrayUtils.toPrimitive((Double[]) data), shape);
        } else if (data instanceof Integer[]) {
            return (TENSOR) IntegerTensor.create(ArrayUtils.toPrimitive(((Integer[]) data)), shape);
        } else if (data instanceof Boolean[]) {
            return (TENSOR) BooleanTensor.create(ArrayUtils.toPrimitive(((Boolean[]) data)), shape);
        } else {
            return (TENSOR) GenericTensor.create(data, shape);
        }
    }

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

    long[] SCALAR_SHAPE = new long[]{};
    long[] SCALAR_STRIDE = new long[]{};
    long[] ONE_BY_ONE_SHAPE = new long[]{1, 1};

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

    default boolean isLengthOne(){
        return getLength() == 1;
    }

    default boolean isScalar() {
        return getRank() == 0;
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
        return getRank() == 1;
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
