package io.improbable.keanu.tensor.generic;

import com.google.common.base.Preconditions;
import io.improbable.keanu.tensor.JVMTensorBroadcast;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.bool.BooleanTensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.BiFunction;

import static io.improbable.keanu.tensor.JVMTensorBroadcast.broadcastIfNeeded;
import static io.improbable.keanu.tensor.TensorShape.convertFromFlatIndexToPermutedFlatIndex;
import static io.improbable.keanu.tensor.TensorShape.getPermutedIndices;
import static io.improbable.keanu.tensor.TensorShape.getReshapeAllowingWildcard;
import static io.improbable.keanu.tensor.TensorShape.getRowFirstStride;
import static java.util.Arrays.copyOf;

public class GenericTensor<T> implements Tensor<T> {

    private T[] data;
    private long[] shape;
    private long[] stride;

    public static <T> GenericTensor<T> createFilled(T data, long[] shape) {
        return new GenericTensor<>(fillArray(shape, data), shape);
    }

    public static <T> GenericTensor<T> create(T... data) {
        return create(data, new long[]{data.length});
    }

    public static <T> GenericTensor<T> create(T[] data, long[] shape) {
        return new GenericTensor<>(data, shape);
    }

    public static <T> GenericTensor<T> scalar(T data) {
        return new GenericTensor<>(data);
    }

    private GenericTensor(T[] data, long[] shape, long[] stride) {
        this.data = data;
        this.shape = shape;
        this.stride = stride;
    }

    private GenericTensor(T[] data, long[] shape) {
        if (TensorShape.getLength(shape) != data.length) {
            throw new IllegalArgumentException("Shape size does not match data length");
        }

        this.data = Arrays.copyOf(data, data.length);
        this.shape = Arrays.copyOf(shape, shape.length);
        this.stride = getRowFirstStride(shape);
    }

    private GenericTensor(T scalar) {
        this.data = (T[]) (new Object[]{scalar});
        this.shape = Tensor.SCALAR_SHAPE;
        this.stride = Tensor.SCALAR_STRIDE;
    }

    private static <T> T[] fillArray(long[] shape, T value) {
        Object[] data = new Object[TensorShape.getLengthAsInt(shape)];
        Arrays.fill(data, value);
        return (T[]) data;
    }

    @Override
    public int getRank() {
        return shape.length;
    }

    @Override
    public long[] getShape() {
        return Arrays.copyOf(shape, shape.length);
    }

    @Override
    public long[] getStride() {
        return stride;
    }

    @Override
    public long getLength() {
        return TensorShape.getLength(shape);
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

        if (!Arrays.equals(shape, that.shape)) return false;
        return Arrays.equals(data, that.data);
    }

    @Override
    public int hashCode() {
        int result = Arrays.hashCode(data);
        result = 31 * result + Arrays.hashCode(shape);
        return result;
    }

    @Override
    public FlattenedView<T> getFlattenedView() {
        return new BaseSimpleFlattenedView<>(data);
    }

    @Override
    public BooleanTensor elementwiseEquals(T value) {
        boolean[] result = new boolean[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = data[i].equals(value);
        }
        return BooleanTensor.create(result, shape);
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
    public GenericTensor<T> reshape(long... newShape) {
        return new GenericTensor<>(Arrays.copyOf(data, data.length), getReshapeAllowingWildcard(shape, data.length, newShape));
    }

    @Override
    public GenericTensor<T> permute(int... rearrange) {
        Preconditions.checkArgument(rearrange.length == shape.length);
        long[] resultShape = getPermutedIndices(shape, rearrange);
        long[] resultStride = getRowFirstStride(resultShape);
        T[] newBuffer = Arrays.copyOf(data, data.length);

        for (int flatIndex = 0; flatIndex < data.length; flatIndex++) {

            int permutedFlatIndex = convertFromFlatIndexToPermutedFlatIndex(
                flatIndex,
                shape, stride,
                resultShape, resultStride,
                rearrange
            );

            newBuffer[permutedFlatIndex] = data[flatIndex];
        }

        return new GenericTensor<>(newBuffer, resultShape);
    }

    @Override
    public GenericTensor<T> slice(int dimension, long index) {
        T[] flat = asFlatArray();
        List<T> tadded = new ArrayList<>();
        for (int i = 0; i < flat.length; i++) {
            long[] indicesOfCurrent = TensorShape.getShapeIndices(shape, stride, i);
            if (indicesOfCurrent[dimension] == index) {
                tadded.add(getValue(indicesOfCurrent));
            }
        }
        long[] taddedShape = Arrays.copyOf(shape, shape.length);
        taddedShape[dimension] = 1;
        return new GenericTensor(tadded.toArray(), taddedShape);
    }

    @Override
    public GenericTensor<T> take(long... index) {
        return scalar(getValue(index));
    }

    public <R> GenericTensor<R> apply(Tensor<T> right,
                                      BiFunction<T, T, R> op) {
        final T[] rightBuffer = right.asFlatArray();
        final long[] rightShape = right.getShape();

        final JVMTensorBroadcast.ResultWrapper result = broadcastIfNeeded(
            data, shape, stride, data.length,
            rightBuffer, rightShape, right.getStride(), rightBuffer.length,
            op, false
        );

        return new GenericTensor<>((R[]) result.outputBuffer, result.outputShape, result.outputStride);
    }

    private void assertIsNumber() {
        if (data.length > 0 && !(data[0] instanceof Number)) {
            throw new IllegalStateException(data[0].getClass().getName() + " cannot be converted to number");
        }
    }
}
