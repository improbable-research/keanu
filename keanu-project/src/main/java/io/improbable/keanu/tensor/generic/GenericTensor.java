package io.improbable.keanu.tensor.generic;

import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import io.improbable.keanu.tensor.JVMTensor;
import io.improbable.keanu.tensor.JVMTensorBroadcast;
import io.improbable.keanu.tensor.ResultWrapper;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.buffer.GenericBuffer;
import org.apache.commons.lang3.ArrayUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.function.BiFunction;

import static io.improbable.keanu.tensor.JVMTensorBroadcast.broadcastIfNeeded;
import static io.improbable.keanu.tensor.TensorShape.convertFromFlatIndexToPermutedFlatIndex;
import static io.improbable.keanu.tensor.TensorShape.getAbsoluteDimension;
import static io.improbable.keanu.tensor.TensorShape.getFlatIndex;
import static io.improbable.keanu.tensor.TensorShape.getPermutedIndices;
import static io.improbable.keanu.tensor.TensorShape.getReshapeAllowingWildcard;
import static io.improbable.keanu.tensor.TensorShape.getRowFirstStride;
import static java.util.Arrays.copyOf;

public class GenericTensor<T> implements Tensor<T, GenericTensor<T>> {

    private static final GenericBuffer.GenericArrayWrapperFactory factory = new GenericBuffer.GenericArrayWrapperFactory();

    private GenericBuffer.PrimitiveGenericWrapper<T> buffer;
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

    private GenericTensor(GenericBuffer.PrimitiveGenericWrapper<T> buffer, long[] shape, long[] stride) {
        this.buffer = buffer;
        this.shape = shape;
        this.stride = stride;
    }

    private GenericTensor(GenericBuffer.PrimitiveGenericWrapper<T> buffer, long[] shape) {
        this.buffer = buffer;
        this.shape = shape;
        this.stride = getRowFirstStride(shape);
    }

    private GenericTensor(T[] buffer, long[] shape, long[] stride) {
        this.buffer = factory.create(buffer);
        this.shape = shape;
        this.stride = stride;
    }

    private GenericTensor(T[] buffer, long[] shape) {
        if (TensorShape.getLength(shape) != buffer.length) {
            throw new IllegalArgumentException("Shape size does not match data length");
        }

        this.buffer = factory.create(Arrays.copyOf(buffer, buffer.length));
        this.shape = Arrays.copyOf(shape, shape.length);
        this.stride = getRowFirstStride(shape);
    }

    private GenericTensor(T scalar) {
        this.buffer = new GenericBuffer.GenericWrapper<>(scalar);
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
        return new GenericTensor<>(buffer.copy(), copyOf(shape, shape.length), copyOf(stride, stride.length));
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        GenericTensor<?> that = (GenericTensor<?>) o;

        if (!Arrays.equals(shape, that.shape)) return false;
        return Arrays.equals(buffer.asArray(), that.buffer.asArray());
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(buffer);
        result = 31 * result + Arrays.hashCode(shape);
        return result;
    }

    @Override
    public FlattenedView<T> getFlattenedView() {
        return new BaseSimpleFlattenedView<>(buffer.asArray());
    }

    @Override
    public BooleanTensor elementwiseEquals(T value) {
        boolean[] result = new boolean[buffer.getLength()];
        for (int i = 0; i < buffer.getLength(); i++) {
            result[i] = buffer.get(i).equals(value);
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
        return buffer.copy().asArray();
    }

    @Override
    public GenericTensor<T> reshape(long... newShape) {
        return new GenericTensor<>(buffer.copy(), getReshapeAllowingWildcard(shape, buffer.getLength(), newShape));
    }

    @Override
    public GenericTensor<T> permute(int... rearrange) {
        Preconditions.checkArgument(rearrange.length == shape.length);
        long[] resultShape = getPermutedIndices(shape, rearrange);
        long[] resultStride = getRowFirstStride(resultShape);
        GenericBuffer.PrimitiveGenericWrapper<T> newBuffer = factory.createNew(buffer.getLength());

        for (int flatIndex = 0; flatIndex < buffer.getLength(); flatIndex++) {

            int permutedFlatIndex = convertFromFlatIndexToPermutedFlatIndex(
                flatIndex,
                shape, stride,
                resultShape, resultStride,
                rearrange
            );

            newBuffer.set(buffer.get(flatIndex), permutedFlatIndex);
        }

        return new GenericTensor<>(newBuffer, resultShape);
    }

    @Override
    public GenericTensor<T> broadcast(long... toShape) {
        int outputLength = TensorShape.getLengthAsInt(toShape);
        long[] outputStride = TensorShape.getRowFirstStride(toShape);
        GenericBuffer.PrimitiveGenericWrapper<T> outputBuffer = factory.createNew(outputLength);

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

            newBuffer[i] = buffer.get(j);
        }

        return new GenericTensor<>(newBuffer, resultShape);
    }

    @Override
    public GenericTensor<T> take(long... index) {
        return scalar(getValue(index));
    }

    @Override
    public List<GenericTensor<T>> split(int dimension, long... splitAtIndices) {
        dimension = getAbsoluteDimension(dimension, getRank());

        if (dimension < 0 || dimension >= shape.length) {
            throw new IllegalArgumentException("Invalid dimension to split on " + dimension);
        }

        int[] moveDimToZero = TensorShape.slideDimension(dimension, 0, shape.length);
        int[] moveZeroToDim = TensorShape.slideDimension(0, dimension, shape.length);

        GenericTensor<T> permutedTensor = this.permute(moveDimToZero);

        GenericBuffer.PrimitiveGenericWrapper<T> rawBuffer = permutedTensor.buffer;

        List<GenericTensor<T>> splitTensor = new ArrayList<>();

        long previousSplitAtIndex = 0;
        int rawBufferPosition = 0;
        for (long splitAtIndex : splitAtIndices) {

            long[] subTensorShape = getShape();
            long subTensorLengthInDimension = splitAtIndex - previousSplitAtIndex;

            if (subTensorLengthInDimension > shape[dimension] || subTensorLengthInDimension <= 0) {
                throw new IllegalArgumentException("Invalid index to split on " + splitAtIndex + " at " + dimension + " for tensor of shape " + Arrays.toString(shape));
            }

            subTensorShape[dimension] = subTensorLengthInDimension;
            int subTensorLength = Ints.checkedCast(TensorShape.getLength(subTensorShape));

            T[] buffer = (T[]) (new Object[subTensorLength]);
            System.arraycopy(rawBuffer.asArray(), rawBufferPosition, buffer, 0, buffer.length);

            long[] subTensorPermutedShape = getPermutedIndices(subTensorShape, moveDimToZero);
            GenericTensor<T> subTensor = GenericTensor.create(buffer, subTensorPermutedShape).permute(moveZeroToDim);
            splitTensor.add(subTensor);

            previousSplitAtIndex = splitAtIndex;
            rawBufferPosition += buffer.length;
        }

        return splitTensor;
    }

    @Override
    public GenericTensor<T> diag() {

        ResultWrapper<T, GenericBuffer.PrimitiveGenericWrapper<T>> result = JVMTensor.diag(getRank(), shape, buffer, factory);

        return new GenericTensor<>(result.outputBuffer, result.outputShape, result.outputStride);
    }

    private static <T> GenericBuffer.PrimitiveGenericWrapper<T> getRawBufferIfJVMTensor(Tensor<T, ?> tensor) {
        if (tensor instanceof GenericTensor) {
            return ((GenericTensor<T>) tensor).buffer;
        } else {
            return factory.create(tensor.asFlatArray());
        }
    }

    public <R> GenericTensor<R> apply(Tensor<T, ?> right,
                                      BiFunction<T, T, R> op) {
        final GenericBuffer.PrimitiveGenericWrapper<T> rightBuffer = getRawBufferIfJVMTensor(right);
        final long[] rightShape = right.getShape();

        final ResultWrapper<R, GenericBuffer.PrimitiveGenericWrapper<R>> result = broadcastIfNeeded(
            factory, buffer, shape, stride, buffer.getLength(),
            rightBuffer, rightShape, right.getStride(), rightBuffer.getLength(),
            op, false
        );

        return new GenericTensor<>(result.outputBuffer, result.outputShape, result.outputStride);
    }

}
