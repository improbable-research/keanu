package io.improbable.keanu.tensor.lng;

import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import io.improbable.keanu.tensor.NumberScalarOperations;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.bool.JVMBooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.JVMDoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.tensor.jvm.JVMFixedPointTensor;
import io.improbable.keanu.tensor.jvm.JVMTensor;
import io.improbable.keanu.tensor.jvm.ResultWrapper;
import io.improbable.keanu.tensor.jvm.buffer.JVMBuffer;

import java.util.Arrays;
import java.util.stream.Collectors;

import static io.improbable.keanu.tensor.TensorShape.getRowFirstStride;

public class JVMLongTensor extends JVMFixedPointTensor<Long, LongTensor, LongBuffer.PrimitiveLongWrapper> implements LongTensor {

    private static final LongBuffer.LongArrayWrapperFactory factory = new LongBuffer.LongArrayWrapperFactory();

    private JVMLongTensor(LongBuffer.PrimitiveLongWrapper buffer, long[] shape, long[] stride) {
        super(buffer, shape, stride);
    }

    private JVMLongTensor(LongBuffer.PrimitiveLongWrapper buffer, long[] shape) {
        super(buffer, shape, getRowFirstStride(shape));
    }

    private JVMLongTensor(ResultWrapper<Long, LongBuffer.PrimitiveLongWrapper> resultWrapper) {
        this(resultWrapper.outputBuffer, resultWrapper.outputShape, resultWrapper.outputStride);
    }

    private JVMLongTensor(long[] data, long[] shape, long[] stride) {
        this(factory.create(data), shape, stride);
    }

    private JVMLongTensor(long[] data, long[] shape) {
        this(factory.create(data), shape);
    }

    private JVMLongTensor(long value) {
        super(new LongBuffer.LongWrapper(value), new long[0], new long[0]);
    }

    @Override
    protected JVMTensor<Long, LongTensor, LongBuffer.PrimitiveLongWrapper> getAsJVMTensor(LongTensor that) {
        return asJVM(that);
    }

    private static JVMLongTensor asJVM(NumberTensor tensor) {
        if (tensor instanceof JVMLongTensor) {
            return ((JVMLongTensor) tensor);
        } else {
            return new JVMLongTensor(factory.create(tensor.asFlatLongArray()), tensor.getShape(), tensor.getStride());
        }
    }

    @Override
    protected JVMBuffer.PrimitiveNumberWrapperFactory<Long, LongBuffer.PrimitiveLongWrapper> getFactory() {
        return factory;
    }

    @Override
    protected NumberScalarOperations<Long> getOperations() {
        return LongScalarOperations.INSTANCE;
    }

    @Override
    protected JVMLongTensor create(LongBuffer.PrimitiveLongWrapper buffer, long[] shape, long[] stride) {
        return new JVMLongTensor(buffer, shape, stride);
    }

    @Override
    protected LongTensor set(LongBuffer.PrimitiveLongWrapper buffer, long[] shape, long[] stride) {
        this.buffer = buffer;
        this.shape = shape;
        this.stride = stride;
        return this;
    }

    public static JVMLongTensor scalar(long scalarValue) {
        return new JVMLongTensor(scalarValue);
    }

    public static JVMLongTensor create(long[] values, long... shape) {
        if (values.length != TensorShape.getLength(shape)) {
            throw new IllegalArgumentException("Shape " + Arrays.toString(shape) + " does not match buffer size " + values.length);
        }
        return new JVMLongTensor(values, shape);
    }

    public static JVMLongTensor create(long value, long... shape) {
        final int length = TensorShape.getLengthAsInt(shape);

        if (length > 1) {
            final long[] buffer = new long[length];
            if (value != 0) {
                Arrays.fill(buffer, value);
            }

            return new JVMLongTensor(buffer, shape);
        } else {
            return new JVMLongTensor(new LongBuffer.LongWrapper(value), shape);
        }
    }

    public static JVMLongTensor ones(long... shape) {
        return create(1, shape);
    }

    public static JVMLongTensor zeros(long... shape) {
        return create(0, shape);
    }

    public static JVMLongTensor eye(long n) {

        if (n == 1) {
            return create(1L, 1, 1);
        } else {

            long[] buffer = new long[Ints.checkedCast(n * n)];
            int nInt = Ints.checkedCast(n);
            for (int i = 0; i < n; i++) {
                buffer[i * nInt + i] = 1;
            }
            return new JVMLongTensor(buffer, new long[]{n, n});
        }
    }

    public static JVMLongTensor arange(long start, long end) {
        return arange(start, end, 1);
    }

    public static JVMLongTensor arange(long start, long end, long stepSize) {
        Preconditions.checkArgument(stepSize != 0);
        int steps = (int) ((end - start) / stepSize);

        return linearBufferCreate(start, steps, stepSize);
    }

    private static JVMLongTensor linearBufferCreate(long start, int numberOfPoints, long stepSize) {
        Preconditions.checkArgument(numberOfPoints > 0);
        long[] buffer = new long[numberOfPoints];

        long currentValue = start;
        for (int i = 0; i < buffer.length; i++, currentValue += stepSize) {
            buffer[i] = currentValue;
        }

        return new JVMLongTensor(buffer, new long[]{buffer.length});
    }

    public static JVMLongTensor concat(int dimension, LongTensor... toConcat) {
        return new JVMLongTensor(
            JVMTensor.concat(factory, toConcat, dimension,
                Arrays.stream(toConcat)
                    .map(tensor -> asJVM(tensor).buffer)
                    .collect(Collectors.toList())
            ));
    }

    @Override
    public BooleanTensor toBoolean() {
        return new JVMBooleanTensor(buffer.equal(1L), getShape(), getStride());
    }

    @Override
    public DoubleTensor toDouble() {
        return JVMDoubleTensor.create(buffer.asDoubleArray(), Arrays.copyOf(shape, shape.length));
    }

    @Override
    public IntegerTensor toInteger() {
        return IntegerTensor.create(buffer.asIntegerArray(), Arrays.copyOf(shape, shape.length));
    }

    @Override
    public LongTensor toLong() {
        return duplicate();
    }

    @Override
    public LongTensor matrixMultiply(LongTensor that) {
        throw new UnsupportedOperationException(
            "Long mmul is not currently supported. Convert to double, matrix multiply, and then convert back to long."
        );
    }

    @Override
    public double[] asFlatDoubleArray() {
        return buffer.asDoubleArray();
    }

    @Override
    public int[] asFlatIntegerArray() {
        return buffer.asIntegerArray();
    }

    @Override
    public long[] asFlatLongArray() {
        return buffer.copy().asLongArray();
    }

    @Override
    public Long[] asFlatArray() {
        return buffer.asArray();
    }

    @Override
    public LongTensor min() {
        long result = Long.MAX_VALUE;
        for (int i = 0; i < buffer.getLength(); i++) {
            result = Math.min(result, buffer.get(i));
        }
        return new JVMLongTensor(result);
    }

    @Override
    public LongTensor minInPlace(LongTensor that) {
        return broadcastableBinaryOpWithAutoBroadcastInPlace(Math::min, getAsJVMTensor(that));
    }

    @Override
    public LongTensor max() {
        long result = Long.MIN_VALUE;
        for (int i = 0; i < buffer.getLength(); i++) {
            result = Math.max(result, buffer.get(i));
        }
        return new JVMLongTensor(result);
    }

    @Override
    public LongTensor maxInPlace(LongTensor that) {
        return broadcastableBinaryOpWithAutoBroadcastInPlace(Math::max, getAsJVMTensor(that));
    }

    @Override
    public LongTensor signInPlace() {
        return applyInPlace(v -> v > 0 ? 1L : v < 0 ? -1 : 0);
    }

    @Override
    public LongTensor absInPlace() {
        return applyInPlace(Math::abs);
    }

    @Override
    public LongTensor unaryMinusInPlace() {
        buffer.apply((v) -> -v);
        return this;
    }

    @Override
    public LongTensor modInPlace(Long that) {
        buffer.apply(v -> v % that);
        return this;
    }

    @Override
    public LongTensor modInPlace(LongTensor that) {
        return broadcastableBinaryOpWithAutoBroadcastInPlace((l, r) -> l % r, getAsJVMTensor(that));
    }

    @Override
    public LongTensor setWithMaskInPlace(LongTensor mask, final Long value) {
        return broadcastableBinaryOpWithAutoBroadcastInPlace((l, r) -> r == 1L ? value : l, getAsJVMTensor(mask));
    }

    @Override
    public BooleanTensor equalsWithinEpsilon(LongTensor that, Long epsilon) {
        return broadcastableBinaryOpToBooleanWithAutoBroadcast(
            (l, r) -> Math.abs(l - r) <= epsilon, getAsJVMTensor(that)
        );
    }

}
