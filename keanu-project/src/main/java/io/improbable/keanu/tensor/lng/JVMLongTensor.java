package io.improbable.keanu.tensor.lng;

import io.improbable.keanu.tensor.NumberScalarOperations;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.bool.JVMBooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.JVMDoubleTensorFactory;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.tensor.jvm.JVMFixedPointTensor;
import io.improbable.keanu.tensor.jvm.JVMTensor;
import io.improbable.keanu.tensor.jvm.ResultWrapper;
import io.improbable.keanu.tensor.jvm.buffer.JVMBuffer;

import java.util.Arrays;

import static io.improbable.keanu.tensor.TensorShape.getRowFirstStride;

public class JVMLongTensor extends JVMFixedPointTensor<Long, LongTensor, LongBuffer.PrimitiveLongWrapper> implements LongTensor {

    static final LongBuffer.LongArrayWrapperFactory factory = new LongBuffer.LongArrayWrapperFactory();

    JVMLongTensor(LongBuffer.PrimitiveLongWrapper buffer, long[] shape, long[] stride) {
        super(buffer, shape, stride);
    }

    JVMLongTensor(LongBuffer.PrimitiveLongWrapper buffer, long[] shape) {
        super(buffer, shape, getRowFirstStride(shape));
    }

    JVMLongTensor(ResultWrapper<Long, LongBuffer.PrimitiveLongWrapper> resultWrapper) {
        this(resultWrapper.outputBuffer, resultWrapper.outputShape, resultWrapper.outputStride);
    }

    JVMLongTensor(long[] data, long[] shape, long[] stride) {
        this(factory.create(data), shape, stride);
    }

    JVMLongTensor(long[] data, long[] shape) {
        this(factory.create(data), shape);
    }

    JVMLongTensor(long value) {
        super(new LongBuffer.LongWrapper(value), new long[0], new long[0]);
    }

    @Override
    protected JVMTensor<Long, LongTensor, LongBuffer.PrimitiveLongWrapper> getAsJVMTensor(LongTensor that) {
        return asJVM(that);
    }

    static JVMLongTensor asJVM(NumberTensor tensor) {
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

    @Override
    public BooleanTensor toBoolean() {
        return JVMBooleanTensor.create(buffer.equal(1L).asBooleanArray(), getShape());
    }

    @Override
    public DoubleTensor toDouble() {
        return JVMDoubleTensorFactory.INSTANCE.create(buffer.asDoubleArray(), Arrays.copyOf(shape, shape.length));
    }

    @Override
    public IntegerTensor toInteger() {
        return IntegerTensor.create(buffer.asIntegerArray(), Arrays.copyOf(shape, shape.length));
    }

    @Override
    public LongTensor toLong() {
        return this;
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
