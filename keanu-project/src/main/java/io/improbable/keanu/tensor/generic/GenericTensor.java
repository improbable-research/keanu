package io.improbable.keanu.tensor.generic;

import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import io.improbable.keanu.tensor.JVMTensorBroadcast;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import org.apache.commons.lang3.ArrayUtils;

import java.util.Arrays;
import java.util.List;
import java.util.function.BiFunction;

import static io.improbable.keanu.tensor.JVMTensorBroadcast.broadcastIfNeeded;
import static io.improbable.keanu.tensor.TensorShape.convertFromFlatIndexToPermutedFlatIndex;
import static io.improbable.keanu.tensor.TensorShape.getFlatIndex;
import static io.improbable.keanu.tensor.TensorShape.getPermutedIndices;
import static io.improbable.keanu.tensor.TensorShape.getReshapeAllowingWildcard;
import static io.improbable.keanu.tensor.TensorShape.getRowFirstStride;
import static java.util.Arrays.copyOf;

public class GenericTensor<T> implements Tensor<T, GenericTensor<T>> {

    private T[] buffer;
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

    private GenericTensor(T[] buffer, long[] shape, long[] stride) {
        this.buffer = buffer;
        this.shape = shape;
        this.stride = stride;
    }

    private GenericTensor(T[] buffer, long[] shape) {
        if (TensorShape.getLength(shape) != buffer.length) {
            throw new IllegalArgumentException("Shape size does not match data length");
        }

        this.buffer = Arrays.copyOf(buffer, buffer.length);
        this.shape = Arrays.copyOf(shape, shape.length);
        this.stride = getRowFirstStride(shape);
    }

    private GenericTensor(T scalar) {
        this.buffer = (T[]) (new Object[]{scalar});
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
        return Arrays.copyOf(stride, stride.length);
    }

    @Override
    public long getLength() {
        return TensorShape.getLength(shape);
    }

    @Override
    public GenericTensor<T> duplicate() {
        return new GenericTensor<>(copyOf(buffer, buffer.length), copyOf(shape, shape.length));
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        GenericTensor<?> that = (GenericTensor<?>) o;

        if (!Arrays.equals(shape, that.shape)) return false;
        return Arrays.equals(buffer, that.buffer);
    }

    @Override
    public int hashCode() {
        int result = Arrays.hashCode(buffer);
        result = 31 * result + Arrays.hashCode(shape);
        return result;
    }

    @Override
    public FlattenedView<T> getFlattenedView() {
        return new BaseSimpleFlattenedView<>(buffer);
    }

    @Override
    public BooleanTensor elementwiseEquals(T value) {
        boolean[] result = new boolean[buffer.length];
        for (int i = 0; i < buffer.length; i++) {
            result[i] = buffer[i].equals(value);
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
    public T[] asFlatArray() {
        return Arrays.copyOf(buffer, buffer.length);
    }

    @Override
    public GenericTensor<T> reshape(long... newShape) {
        return new GenericTensor<>(Arrays.copyOf(buffer, buffer.length), getReshapeAllowingWildcard(shape, buffer.length, newShape));
    }

    @Override
    public GenericTensor<T> permute(int... rearrange) {
        Preconditions.checkArgument(rearrange.length == shape.length);
        long[] resultShape = getPermutedIndices(shape, rearrange);
        long[] resultStride = getRowFirstStride(resultShape);
        T[] newBuffer = Arrays.copyOf(buffer, buffer.length);

        for (int flatIndex = 0; flatIndex < buffer.length; flatIndex++) {

            int permutedFlatIndex = convertFromFlatIndexToPermutedFlatIndex(
                flatIndex,
                shape, stride,
                resultShape, resultStride,
                rearrange
            );

            newBuffer[permutedFlatIndex] = buffer[flatIndex];
        }

        return new GenericTensor<>(newBuffer, resultShape);
    }

    @Override
    public GenericTensor<T> broadcast(long... toShape) {
        int outputLength = TensorShape.getLengthAsInt(toShape);
        long[] outputStride = TensorShape.getRowFirstStride(toShape);
        T[] outputBuffer = (T[]) (new Object[outputLength]);

        JVMTensorBroadcast.broadcast(buffer, shape, stride, outputBuffer, outputStride);

        return new GenericTensor<>(outputBuffer, toShape, outputStride);
    }

    @Override
    public GenericTensor<T> slice(int dimension, long index) {
        Preconditions.checkArgument(dimension < shape.length && index < shape[dimension]);
        long[] resultShape = ArrayUtils.remove(shape, dimension);
        long[] resultStride = getRowFirstStride(resultShape);
        T[] newBuffer = (T[]) new Object[Ints.checkedCast(TensorShape.getLength(resultShape))];

        for (int i = 0; i < newBuffer.length; i++) {

            long[] shapeIndices = ArrayUtils.insert(dimension, TensorShape.getShapeIndices(resultShape, resultStride, i), index);

            int j = Ints.checkedCast(getFlatIndex(shape, stride, shapeIndices));

            newBuffer[i] = buffer[j];
        }

        return new GenericTensor<>(newBuffer, resultShape);
    }

    @Override
    public GenericTensor<T> take(long... index) {
        return scalar(getValue(index));
    }

    @Override
    public List<GenericTensor<T>> split(int dimension, long... splitAtIndices) {
        return null;
    }

    @Override
    public GenericTensor<T> diag() {

        T[] newBuffer;
        long[] newShape;
        if (getRank() == 1) {
            int n = buffer.length;
            newBuffer = (T[]) (new Object[Ints.checkedCast((long) n * (long) n)]);
            for (int i = 0; i < n; i++) {
                newBuffer[i * n + i] = buffer[i];
            }
            newShape = new long[]{n, n};
        } else if (getRank() == 2 && shape[0] == shape[1]) {
            int n = Ints.checkedCast(shape[0]);
            newBuffer = (T[]) (new Object[n]);
            for (int i = 0; i < n; i++) {
                newBuffer[i] = buffer[i * n + i];
            }
            newShape = new long[]{n};
        } else {
            throw new IllegalArgumentException("Diag is only valid for vectors or square matrices");
        }

        return new GenericTensor<>(newBuffer, newShape);
    }

    public <R> GenericTensor<R> apply(Tensor<T, ?> right,
                                      BiFunction<T, T, R> op) {
        final T[] rightBuffer = right.asFlatArray();
        final long[] rightShape = right.getShape();

        final JVMTensorBroadcast.ResultWrapper<T[]> result = broadcastIfNeeded(
            buffer, shape, stride, buffer.length,
            rightBuffer, rightShape, right.getStride(), rightBuffer.length,
            op, false
        );

        return new GenericTensor<>((R[]) result.outputBuffer, result.outputShape, result.outputStride);
    }

    private void assertIsNumber() {
        if (buffer.length > 0 && !(buffer[0] instanceof Number)) {
            throw new IllegalStateException(buffer[0].getClass().getName() + " cannot be converted to number");
        }
    }
}
