package io.improbable.keanu.tensor.bool;

import com.google.common.base.Preconditions;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.tensor.jvm.JVMTensor;
import io.improbable.keanu.tensor.jvm.ResultWrapper;
import io.improbable.keanu.tensor.jvm.buffer.JVMBuffer;
import org.apache.commons.lang3.ArrayUtils;

import java.util.Arrays;
import java.util.stream.Collectors;

import static io.improbable.keanu.tensor.TensorShape.getRowFirstStride;
import static io.improbable.keanu.tensor.bool.BroadcastableBooleanOperations.AND;
import static io.improbable.keanu.tensor.bool.BroadcastableBooleanOperations.OR;
import static io.improbable.keanu.tensor.bool.BroadcastableBooleanOperations.XOR;
import static java.util.Arrays.copyOf;

public class JVMBooleanTensor extends JVMTensor<Boolean, BooleanTensor, BooleanBuffer.PrimitiveBooleanWrapper> implements BooleanTensor {

    private static final BooleanBuffer.BooleanArrayWrapperFactory factory = BooleanBuffer.factory;

    public JVMBooleanTensor(BooleanBuffer.PrimitiveBooleanWrapper buffer, long[] shape, long[] stride) {
        super(buffer, shape, stride);
    }

    public JVMBooleanTensor(BooleanBuffer.PrimitiveBooleanWrapper buffer, long[] shape) {
        super(buffer, shape, getRowFirstStride(shape));
    }

    /**
     * @param buffer tensor buffer used c ordering
     * @param shape  desired shape of tensor
     */
    private JVMBooleanTensor(boolean[] buffer, long[] shape) {
        this(factory.create(buffer), shape);
    }

    private JVMBooleanTensor(boolean[] buffer, long[] shape, long[] stride) {
        this(factory.create(buffer), shape, stride);
    }

    /**
     * @param constant constant boolean value to fill shape
     */
    private JVMBooleanTensor(boolean constant) {
        super(new BooleanBuffer.BooleanWrapper(constant), new long[0], new long[0]);
    }

    private JVMBooleanTensor(ResultWrapper<Boolean, BooleanBuffer.PrimitiveBooleanWrapper> wrapper) {
        this(wrapper.outputBuffer, wrapper.outputShape, wrapper.outputStride);
    }

    @Override
    protected JVMBooleanTensor create(BooleanBuffer.PrimitiveBooleanWrapper buffer, long[] shape, long[] stride) {
        return new JVMBooleanTensor(buffer, shape, stride);
    }

    @Override
    protected JVMBooleanTensor set(BooleanBuffer.PrimitiveBooleanWrapper buffer, long[] shape, long[] stride) {
        this.buffer = buffer;
        this.shape = shape;
        this.stride = stride;
        return this;
    }

    @Override
    protected JVMBuffer.ArrayWrapperFactory<Boolean, BooleanBuffer.PrimitiveBooleanWrapper> getFactory() {
        return factory;
    }

    public static JVMBooleanTensor scalar(boolean scalarValue) {
        return new JVMBooleanTensor(scalarValue);
    }

    public static JVMBooleanTensor create(boolean[] values, long... shape) {
        Preconditions.checkArgument(
            TensorShape.getLength(shape) == values.length,
            "Shape " + Arrays.toString(shape) + " does not match data length " + values.length
        );
        return new JVMBooleanTensor(values, shape);
    }

    public static JVMBooleanTensor create(boolean value, long... shape) {
        boolean[] buffer = new boolean[TensorShape.getLengthAsInt(shape)];

        if (value) {
            Arrays.fill(buffer, value);
        }

        return new JVMBooleanTensor(buffer, shape);
    }

    public static BooleanTensor concat(int dimension, BooleanTensor... toConcat) {
        return new JVMBooleanTensor(
            JVMTensor.concat(factory, toConcat, dimension,
                Arrays.stream(toConcat)
                    .map(tensor -> getAsJVMTensor(tensor).buffer)
                    .collect(Collectors.toList())
            ));
    }

    private static JVMBooleanTensor getAsJVMTensor(BooleanTensor tensor) {
        if (tensor instanceof JVMBooleanTensor) {
            return ((JVMBooleanTensor) tensor);
        } else {
            return new JVMBooleanTensor(factory.create(tensor.asFlatBooleanArray()), tensor.getShape(), tensor.getStride());
        }
    }

    @Override
    public BooleanTensor andInPlace(BooleanTensor that) {
        return broadcastableBinaryOpWithAutoBroadcast(AND, getAsJVMTensor(that));
    }

    @Override
    public BooleanTensor andInPlace(boolean that) {
        if (!that) {
            buffer.applyRight((l, r) -> r, false);
        }
        return this;
    }

    @Override
    public BooleanTensor orInPlace(BooleanTensor that) {
        return broadcastableBinaryOpWithAutoBroadcast(OR, getAsJVMTensor(that));
    }

    @Override
    public BooleanTensor orInPlace(boolean that) {
        if (that) {
            buffer.applyRight((l, r) -> r, true);
        }
        return this;
    }

    @Override
    public BooleanTensor xorInPlace(BooleanTensor that) {
        return broadcastableBinaryOpWithAutoBroadcast(XOR, getAsJVMTensor(that));
    }

    @Override
    public BooleanTensor notInPlace() {
        buffer.apply(v -> !v);
        return this;
    }

    @Override
    public BooleanTensor allTrue() {
        for (int i = 0; i < buffer.getLength(); i++) {
            if (!buffer.get(i)) {
                return new JVMBooleanTensor(false);
            }
        }
        return new JVMBooleanTensor(true);
    }

    @Override
    public BooleanTensor allFalse() {
        for (int i = 0; i < buffer.getLength(); i++) {
            if (buffer.get(i)) {
                return new JVMBooleanTensor(false);
            }
        }
        return new JVMBooleanTensor(true);
    }

    @Override
    public BooleanTensor anyTrue() {
        return allFalse().notInPlace();
    }

    @Override
    public BooleanTensor anyFalse() {
        return allTrue().notInPlace();
    }

    @Override
    public DoubleTensor toDoubleMask() {
        double[] doubles = asFlatDoubleArray();
        return DoubleTensor.create(doubles, copyOf(shape, shape.length));
    }

    @Override
    public IntegerTensor toIntegerMask() {
        int[] doubles = asFlatIntegerArray();
        return IntegerTensor.create(doubles, copyOf(shape, shape.length));
    }

    @Override
    public BooleanTensor elementwiseEquals(Boolean value) {
        return value ? duplicate() : not();
    }

    @Override
    public BooleanTensor elementwiseEquals(BooleanTensor that) {
        return xor(that).notInPlace();
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
    public Boolean[] asFlatArray() {
        return ArrayUtils.toObject(asFlatBooleanArray());
    }

    @Override
    public boolean[] asFlatBooleanArray() {
        return buffer.copy().asBooleanArray();
    }

}
