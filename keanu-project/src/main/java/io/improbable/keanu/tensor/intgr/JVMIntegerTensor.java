package io.improbable.keanu.tensor.intgr;

import io.improbable.keanu.tensor.FixedPointScalarOperations;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.bool.JVMBooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.JVMDoubleTensorFactory;
import io.improbable.keanu.tensor.jvm.JVMFixedPointTensor;
import io.improbable.keanu.tensor.jvm.JVMTensor;
import io.improbable.keanu.tensor.jvm.ResultWrapper;
import io.improbable.keanu.tensor.jvm.buffer.JVMBuffer;
import io.improbable.keanu.tensor.lng.LongTensor;

import java.util.Arrays;

import static io.improbable.keanu.tensor.TensorShape.getRowFirstStride;

public class JVMIntegerTensor extends JVMFixedPointTensor<Integer, IntegerTensor, IntegerBuffer.PrimitiveIntegerWrapper> implements IntegerTensor {

    static final IntegerBuffer.IntegerArrayWrapperFactory factory = new IntegerBuffer.IntegerArrayWrapperFactory();

    JVMIntegerTensor(IntegerBuffer.PrimitiveIntegerWrapper buffer, long[] shape, long[] stride) {
        super(buffer, shape, stride);
    }

    JVMIntegerTensor(IntegerBuffer.PrimitiveIntegerWrapper buffer, long[] shape) {
        super(buffer, shape, getRowFirstStride(shape));
    }

    JVMIntegerTensor(ResultWrapper<Integer, IntegerBuffer.PrimitiveIntegerWrapper> resultWrapper) {
        this(resultWrapper.outputBuffer, resultWrapper.outputShape, resultWrapper.outputStride);
    }

    JVMIntegerTensor(int[] data, long[] shape, long[] stride) {
        this(factory.create(data), shape, stride);
    }

    JVMIntegerTensor(int[] data, long[] shape) {
        this(factory.create(data), shape);
    }

    JVMIntegerTensor(int value) {
        super(new IntegerBuffer.IntegerWrapper(value), new long[0], new long[0]);
    }

    @Override
    protected JVMTensor<Integer, IntegerTensor, IntegerBuffer.PrimitiveIntegerWrapper> getAsJVMTensor(IntegerTensor that) {
        return asJVM(that);
    }

    static JVMIntegerTensor asJVM(NumberTensor tensor) {
        if (tensor instanceof JVMIntegerTensor) {
            return ((JVMIntegerTensor) tensor);
        } else {
            return new JVMIntegerTensor(factory.create(tensor.asFlatIntegerArray()), tensor.getShape(), tensor.getStride());
        }
    }

    @Override
    protected JVMBuffer.PrimitiveNumberWrapperFactory<Integer, IntegerBuffer.PrimitiveIntegerWrapper> getFactory() {
        return factory;
    }

    @Override
    protected FixedPointScalarOperations<Integer> getOperations() {
        return IntegerScalarOperations.INSTANCE;
    }

    @Override
    protected JVMIntegerTensor create(IntegerBuffer.PrimitiveIntegerWrapper buffer, long[] shape, long[] stride) {
        return new JVMIntegerTensor(buffer, shape, stride);
    }

    @Override
    protected IntegerTensor set(IntegerBuffer.PrimitiveIntegerWrapper buffer, long[] shape, long[] stride) {
        this.buffer = buffer;
        this.shape = shape;
        this.stride = stride;
        return this;
    }

    @Override
    public BooleanTensor toBoolean() {
        return JVMBooleanTensor.create(buffer.equal(1).asBooleanArray(), getShape());
    }

    @Override
    public DoubleTensor toDouble() {
        return JVMDoubleTensorFactory.INSTANCE.create(buffer.asDoubleArray(), Arrays.copyOf(shape, shape.length));
    }

    @Override
    public LongTensor toLong() {
        return LongTensor.create(buffer.asLongArray(), Arrays.copyOf(shape, shape.length));
    }

    @Override
    public IntegerTensor toInteger() {
        return this;
    }

    @Override
    public IntegerTensor matrixMultiply(IntegerTensor that, boolean transposeLeft, boolean transposeRight) {
        throw new UnsupportedOperationException(
            "Integer mmul is not currently supported. Convert to double, matrix multiply, and then convert back to long."
        );
    }

    @Override
    public double[] asFlatDoubleArray() {
        return buffer.asDoubleArray();
    }

    @Override
    public int[] asFlatIntegerArray() {
        return buffer.copy().asIntegerArray();
    }

    @Override
    public long[] asFlatLongArray() {
        return buffer.asLongArray();
    }

    @Override
    public Integer[] asFlatArray() {
        return buffer.asArray();
    }

    @Override
    public IntegerTensor setWithMaskInPlace(IntegerTensor mask, final Integer value) {
        return broadcastableBinaryOpWithAutoBroadcastInPlace((l, r) -> r == 1L ? value : l, getAsJVMTensor(mask));
    }

}
