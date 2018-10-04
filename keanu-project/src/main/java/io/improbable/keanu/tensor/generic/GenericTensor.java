package io.improbable.keanu.tensor.generic;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import static java.util.Arrays.copyOf;

import static io.improbable.keanu.tensor.TensorShape.getFlatIndex;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;

public class GenericTensor<T> implements Tensor<T> {

    private T[] data;
    private int[] shape;
    private int[] stride;

    public GenericTensor(T[] data, int[] shape) {
        this.data = Arrays.copyOf(data, data.length);
        this.shape = Arrays.copyOf(shape, shape.length);
        this.stride = TensorShape.getRowFirstStride(shape);

        if (getLength() != data.length) {
            throw new IllegalArgumentException("Shape size does not match data length");
        }
    }

    public GenericTensor(T data, int[] shape) {
        this.data = (T[]) (new Object[(int) TensorShape.getLength(shape)]);
        Arrays.fill(this.data, data);

        this.shape = Arrays.copyOf(shape, shape.length);
        this.stride = TensorShape.getRowFirstStride(shape);
    }

    public GenericTensor(T scalar) {
        this.data = (T[]) (new Object[]{scalar});
        this.shape = Tensor.SCALAR_SHAPE;
        this.stride = Tensor.SCALAR_STRIDE;
    }

    /**
     * @param shape placeholder shape
     */
    public GenericTensor(int[] shape) {
        this.data = null;
        this.shape = Arrays.copyOf(shape, shape.length);
        this.stride = TensorShape.getRowFirstStride(shape);
    }

    @Override
    public int getRank() {
        return shape.length;
    }

    @Override
    public int[] getShape() {
        return Arrays.copyOf(shape, shape.length);
    }

    @Override
    public long getLength() {
        return TensorShape.getLength(shape);
    }

    @Override
    public boolean isShapePlaceholder() {
        return data == null;
    }

    @Override
    public T getValue(int... index) {
        return data[getFlatIndex(shape, stride, index)];
    }

    @Override
    public GenericTensor<T> setValue(T value, int... index) {
        data[getFlatIndex(shape, stride, index)] = value;
        return this;
    }

    public boolean isNull() {
        return this.data == null;
    }

    public GenericTensor<T> setWithMaskInPlace(DoubleTensor mask, T value) {
        if (this.getLength() != mask.getLength()) {
            throw new IllegalArgumentException("The lengths of the tensor and mask must match, but got tensor length: " + this.getLength() + ", mask length: " + mask.getLength());
        }

        if (data == null) {
            data = (T[]) new Object[(int) this.getLength()];
        }

        double[] flatArray = mask.asFlatDoubleArray();
        for (int i = 0; i < flatArray.length; i++) {
            if (flatArray[i] == 1.) {
                data[i] = value;
            }
        }

        return this;
    }

    public DoubleTensor equalsMask(T value) {
        return DoubleTensor.create(
            Arrays.stream(data).mapToDouble(d -> Objects.equals(value, d) ? 1. : 0.).toArray(),
            this.shape);
    }

    @Override
    public T scalar() {
        return data[0];
    }

    @Override
    public GenericTensor<T> duplicate() {
        return new GenericTensor<>(copyOf(data, data.length), copyOf(shape, shape.length));
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        GenericTensor<?> that = (GenericTensor<?>) o;

        if (!Arrays.equals(data, that.data)) return false;
        if (!Arrays.equals(shape, that.shape)) return false;
        return Arrays.equals(stride, that.stride);
    }

    @Override
    public int hashCode() {
        int result = Arrays.hashCode(data);
        result = 31 * result + Arrays.hashCode(shape);
        result = 31 * result + Arrays.hashCode(stride);
        return result;
    }

    @Override
    public FlattenedView<T> getFlattenedView() {
        return new BaseSimpleFlattenedView<T>(data);
    }

    private static class BaseSimpleFlattenedView<T> implements FlattenedView<T> {

        T[] data;

        public BaseSimpleFlattenedView(T[] data) {
            this.data = data;
        }

        @Override
        public long size() {
            return data.length;
        }

        @Override
        public T get(long index) {
            if (index > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("Only integer based indexing supported for generic tensors");
            }
            return data[(int) index];
        }

        @Override
        public T getOrScalar(long index) {
            if (data.length == 1) {
                return get(0);
            } else {
                return get(index);
            }
        }

        @Override
        public void set(long index, T value) {
            if (index > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("Only integer based indexing supported for generic tensors");
            }
            data[(int) index] = value;
        }

    }

    @Override
    public double[] asFlatDoubleArray() {

        assertIsNumber();

        double[] doubles = new double[data.length];
        for (int i = 0; i < doubles.length; i++) {
            doubles[i] = ((Number) data[i]).doubleValue();
        }

        return doubles;
    }

    @Override
    public int[] asFlatIntegerArray() {

        assertIsNumber();

        int[] integers = new int[data.length];
        for (int i = 0; i < integers.length; i++) {
            integers[i] = ((Number) data[i]).intValue();
        }

        return integers;
    }

    @Override
    public T[] asFlatArray() {
        return Arrays.copyOf(data, data.length);
    }

    @Override
    public Tensor<T> reshape(int... newShape) {
        if (TensorShape.getLength(shape) != TensorShape.getLength(newShape)) {
            throw new IllegalArgumentException("Cannot reshape a tensor to a shape of different length. Failed to reshape: "
                + Arrays.toString(shape) + " to: " + Arrays.toString(newShape));
        }
        return new GenericTensor<>(data, newShape);
    }

    @Override
    public Tensor<T> slice(int dimension, int index) {
        T[] flat = asFlatArray();
        List<T> tadded = new ArrayList<>();
        for (int i = 0; i < flat.length; i++) {
            int[] indicesOfCurrent = TensorShape.getShapeIndices(shape, stride, i);
            if (indicesOfCurrent[dimension] == index) {
                tadded.add(getValue(indicesOfCurrent));
            }
        }
        int[] taddedShape = Arrays.copyOf(shape, shape.length);
        taddedShape[dimension] = 1;
        return new GenericTensor(tadded.toArray(), taddedShape);
    }

    private void assertIsNumber() {
        if (data.length > 0 && !(data[0] instanceof Number)) {
            throw new IllegalStateException(data[0].getClass().getName() + " cannot be converted to number");
        }
    }
}
