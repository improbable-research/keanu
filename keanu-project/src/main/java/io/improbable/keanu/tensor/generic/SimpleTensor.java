package io.improbable.keanu.tensor.generic;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;

import static io.improbable.keanu.tensor.TensorShape.getFlatIndex;
import static java.util.Arrays.copyOf;

public class SimpleTensor<T> implements Tensor<T> {

    private T[] data;
    private int[] shape;
    private int[] stride;

    public SimpleTensor(T[] data, int[] shape) {
        this.data = data;
        this.shape = shape;
        this.stride = TensorShape.getRowFirstStride(shape);

        if (getLength() != data.length) {
            throw new IllegalArgumentException("Shape size does not match data length");
        }
    }

    public SimpleTensor(T scalar) {
        this.data = (T[]) (new Object[]{scalar});
        this.shape = Tensor.SCALAR_SHAPE;
        this.stride = Tensor.SCALAR_STRIDE;
    }

    /**
     * @param shape placeholder shape
     */
    public SimpleTensor(int[] shape) {
        this.data = null;
        this.shape = shape;
        this.stride = TensorShape.getRowFirstStride(shape);
    }

    @Override
    public int getRank() {
        return shape.length;
    }

    @Override
    public int[] getShape() {
        return shape;
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
    public void setValue(T value, int... index) {
        data[getFlatIndex(shape, stride, index)] = value;
    }

    @Override
    public T scalar() {
        return data[0];
    }

    @Override
    public SimpleTensor<T> duplicate() {
        return new SimpleTensor<>(copyOf(data, data.length), copyOf(shape, shape.length));
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
    public double[] asDoubleArray() {

        assertIsNumber();

        double[] doubles = new double[data.length];
        for (int i = 0; i < doubles.length; i++) {
            doubles[i] = ((Number) data[i]).doubleValue();
        }

        return doubles;
    }

    @Override
    public int[] asIntegerArray() {

        assertIsNumber();

        int[] integers = new int[data.length];
        for (int i = 0; i < integers.length; i++) {
            integers[i] = ((Number) data[i]).intValue();
        }

        return integers;
    }

    @Override
    public T[] asArray() {
        return data;
    }

    private void assertIsNumber() {
        if (data.length > 0 && !(data[0] instanceof Number)) {
            throw new IllegalStateException(data[0].getClass().getName() + " cannot be converted to number");
        }
    }
}
