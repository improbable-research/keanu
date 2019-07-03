package io.improbable.keanu.tensor.generic;

import com.google.common.primitives.Ints;
import io.improbable.keanu.tensor.jvm.JVMTensor;
import io.improbable.keanu.tensor.jvm.ResultWrapper;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.buffer.JVMBuffer;

import java.util.Arrays;
import java.util.function.BiFunction;

import static io.improbable.keanu.tensor.jvm.JVMTensorBroadcast.broadcastIfNeeded;
import static io.improbable.keanu.tensor.TensorShape.getRowFirstStride;
import static java.util.Arrays.copyOf;

public class GenericTensor<T> extends JVMTensor<T, GenericTensor<T>, GenericBuffer.PrimitiveGenericWrapper<T>> implements Tensor<T, GenericTensor<T>> {

    private static final GenericBuffer.GenericArrayWrapperFactory factory = new GenericBuffer.GenericArrayWrapperFactory();

    public static <T> GenericTensor<T> createFilled(T data, long[] shape) {
        return new GenericTensor<>(fillArray(shape, data), shape);
    }

    public static <T> GenericTensor<T> create(T... data) {
        return create(data, new long[]{data.length});
    }

    public static <T> GenericTensor<T> create(T[] data, long[] shape) {
        if (TensorShape.getLength(shape) != data.length) {
            throw new IllegalArgumentException("Shape size does not match data length");
        }
        return new GenericTensor<>(data, shape);
    }

    public static <T> GenericTensor<T> scalar(T data) {
        return new GenericTensor<>(data);
    }

    private GenericTensor(GenericBuffer.PrimitiveGenericWrapper<T> buffer, long[] shape, long[] stride) {
        super(buffer, shape, stride);
    }

    private GenericTensor(GenericBuffer.PrimitiveGenericWrapper<T> buffer, long[] shape) {
        super(buffer, shape, getRowFirstStride(shape));
    }

    private GenericTensor(T[] buffer, long[] shape, long[] stride) {
        super(factory.create(buffer), shape, stride);
    }

    private GenericTensor(T[] buffer, long[] shape) {
        this(buffer, shape, getRowFirstStride(shape));
    }

    private GenericTensor(T scalar) {
        super(new GenericBuffer.GenericWrapper<>(scalar), new long[0], new long[0]);
    }

    private static <T> T[] fillArray(long[] shape, T value) {
        Object[] data = new Object[TensorShape.getLengthAsInt(shape)];
        Arrays.fill(data, value);
        return (T[]) data;
    }

    @Override
    public GenericTensor<T> duplicate() {
        return new GenericTensor<>(buffer.copy(), copyOf(shape, shape.length), copyOf(stride, stride.length));
    }

    @Override
    public FlattenedView<T> getFlattenedView() {
        return new BaseSimpleFlattenedView<>(buffer.asArray());
    }

    @Override
    public BooleanTensor elementwiseEquals(T value) {
        boolean[] result = new boolean[Ints.checkedCast(buffer.getLength())];
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
    protected GenericTensor<T> create(GenericBuffer.PrimitiveGenericWrapper<T> buffer, long[] shape, long[] stride) {
        return new GenericTensor<>(buffer, shape, stride);
    }

    @Override
    protected GenericTensor<T> set(GenericBuffer.PrimitiveGenericWrapper<T> buffer, long[] shape, long[] stride) {
        this.buffer = buffer;
        this.shape = shape;
        this.stride = stride;
        return this;
    }

    @Override
    protected JVMBuffer.ArrayWrapperFactory<T, GenericBuffer.PrimitiveGenericWrapper<T>> getFactory() {
        return factory;
    }

    @Override
    public GenericTensor<T> take(long... index) {
        return scalar(getValue(index));
    }

    private static <T> GenericTensor<T> getRawBufferIfJVMTensor(Tensor<T, ?> tensor) {
        if (tensor instanceof GenericTensor) {
            return ((GenericTensor<T>) tensor);
        } else {
            return new GenericTensor<T>(factory.create(tensor.asFlatArray()), tensor.getShape(), tensor.getStride());
        }
    }

    public <R> GenericTensor<R> apply(Tensor<T, ?> right,
                                      BiFunction<T, T, R> op) {

        final GenericTensor<T> rightTensor = getRawBufferIfJVMTensor(right);

        final ResultWrapper<R, GenericBuffer.PrimitiveGenericWrapper<R>> result = broadcastIfNeeded(
            factory, buffer, shape, stride, buffer.getLength(),
            rightTensor.buffer, rightTensor.shape, rightTensor.stride, rightTensor.buffer.getLength(),
            op, false
        );

        return new GenericTensor<>(result.outputBuffer, result.outputShape, result.outputStride);
    }

}
