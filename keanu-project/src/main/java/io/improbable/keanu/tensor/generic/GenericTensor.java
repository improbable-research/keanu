package io.improbable.keanu.tensor.generic;

import com.google.common.primitives.Ints;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.jvm.JVMTensor;
import io.improbable.keanu.tensor.jvm.ResultWrapper;
import io.improbable.keanu.tensor.jvm.buffer.JVMBuffer;
import org.apache.commons.lang3.ArrayUtils;

import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.function.BiFunction;

import static io.improbable.keanu.tensor.TensorShape.getRowFirstStride;
import static io.improbable.keanu.tensor.jvm.JVMTensorBroadcast.broadcastIfNeeded;
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
        return new BaseSimpleFlattenedView();
    }

    @Override
    public BooleanTensor elementwiseEquals(T value) {
        boolean[] result = new boolean[Ints.checkedCast(buffer.getLength())];
        for (int i = 0; i < buffer.getLength(); i++) {
            result[i] = buffer.get(i).equals(value);
        }
        return BooleanTensor.create(result, shape);
    }

    @Override
    public BooleanTensor elementwiseEquals(GenericTensor<T> that) {
        if (isScalar()) {
            return that.elementwiseEquals(this.scalar());
        } else if (that.isScalar()) {
            return elementwiseEquals(that.scalar());
        } else {

            GenericTensor<Boolean> resultTensor = apply(that, (l, r) -> l.equals(r));
            if (resultTensor.getLength() > 0) {
                boolean[] result = ArrayUtils.toPrimitive(resultTensor.asFlatArray());
                return BooleanTensor.create(result, getShape());
            } else {
                return BooleanTensor.create();
            }
        }
    }

    private class BaseSimpleFlattenedView implements FlattenedView<T> {

        @Override
        public long size() {
            return buffer.getLength();
        }

        @Override
        public T get(long index) {
            return buffer.get(index);
        }

        @Override
        public T getOrScalar(long index) {
            if (buffer.getLength() == 1) {
                return get(0);
            } else {
                return get(index);
            }
        }

        @Override
        public void set(long index, T value) {
            buffer.set(value, index);
        }

    }

    @Override
    public T[] asFlatArray() {
        if (buffer.getLength() == 0) {
            return (T[]) (new Object[0]);
        } else {
            int length = Ints.checkedCast(buffer.getLength());
            T[] typed = (T[]) Array.newInstance(buffer.get(0).getClass(), Ints.checkedCast(length));
            System.arraycopy(buffer.asArray(), 0, typed, 0, length);
            return typed;
        }
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
