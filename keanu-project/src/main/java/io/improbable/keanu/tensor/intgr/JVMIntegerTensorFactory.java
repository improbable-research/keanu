package io.improbable.keanu.tensor.intgr;

import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.jvm.JVMTensor;

import java.util.Arrays;
import java.util.stream.Collectors;


public class JVMIntegerTensorFactory implements IntegerTensorFactory {

    public static final JVMIntegerTensorFactory INSTANCE = new JVMIntegerTensorFactory();

    @Override
    public IntegerTensor create(long[] values, long[] shape) {
        return create(toIntegers(values), shape);
    }

    @Override
    public IntegerTensor create(int[] values) {
        return create(values, values.length);
    }

    @Override
    public IntegerTensor create(long[] values) {
        return create(toIntegers(values), values.length);
    }

    @Override
    public JVMIntegerTensor create(int[] values, long... shape) {
        if (values.length != TensorShape.getLength(shape)) {
            throw new IllegalArgumentException("Shape " + Arrays.toString(shape) + " does not match buffer size " + values.length);
        }
        return new JVMIntegerTensor(values, shape);
    }

    @Override
    public JVMIntegerTensor create(int value, long... shape) {
        final int length = TensorShape.getLengthAsInt(shape);

        if (length > 1) {
            final int[] buffer = new int[length];
            if (value != 0) {
                Arrays.fill(buffer, value);
            }

            return new JVMIntegerTensor(buffer, shape);
        } else {
            return new JVMIntegerTensor(new IntegerBuffer.IntegerWrapper(value), shape);
        }
    }

    @Override
    public IntegerTensor create(long value, long[] shape) {
        return create(Ints.checkedCast(value), shape);
    }

    @Override
    public IntegerTensor eye(long n) {
        if (n == 1) {
            return create(1, new long[]{1, 1});
        } else {

            int[] buffer = new int[Ints.checkedCast(n * n)];
            int nInt = Ints.checkedCast(n);
            for (int i = 0; i < n; i++) {
                buffer[i * nInt + i] = 1;
            }
            return new JVMIntegerTensor(buffer, new long[]{n, n});
        }
    }

    private int[] toIntegers(long[] longs) {
        final int[] ints = new int[longs.length];
        for (int i = 0; i < ints.length; i++) {
            ints[i] = Ints.checkedCast(longs[i]);
        }
        return ints;
    }

    public JVMIntegerTensor scalar(int scalarValue) {
        return new JVMIntegerTensor(scalarValue);
    }

    @Override
    public IntegerTensor scalar(long scalarValue) {
        return scalar(Ints.checkedCast(scalarValue));
    }

    @Override
    public JVMIntegerTensor ones(long... shape) {
        return create(1, shape);
    }

    @Override
    public JVMIntegerTensor zeros(long... shape) {
        return create(0, shape);
    }

    @Override
    public JVMIntegerTensor arange(Integer start, Integer end) {
        return arange(start, end, 1);
    }

    @Override
    public JVMIntegerTensor arange(Integer start, Integer end, Integer stepSize) {
        Preconditions.checkArgument(stepSize != 0);
        int steps = ((end - start) / stepSize);

        return linearBufferCreate(start, steps, stepSize);
    }

    private static JVMIntegerTensor linearBufferCreate(int start, int numberOfPoints, int stepSize) {
        Preconditions.checkArgument(numberOfPoints > 0);
        int[] buffer = new int[numberOfPoints];

        int currentValue = start;
        for (int i = 0; i < buffer.length; i++, currentValue += stepSize) {
            buffer[i] = currentValue;
        }

        return new JVMIntegerTensor(buffer, new long[]{buffer.length});
    }

    @Override
    public JVMIntegerTensor concat(int dimension, IntegerTensor... toConcat) {
        return new JVMIntegerTensor(
            JVMTensor.concat(JVMIntegerTensor.factory, toConcat, dimension,
                Arrays.stream(toConcat)
                    .map(tensor -> JVMIntegerTensor.asJVM(tensor).getBuffer())
                    .collect(Collectors.toList())
            ));
    }
}
