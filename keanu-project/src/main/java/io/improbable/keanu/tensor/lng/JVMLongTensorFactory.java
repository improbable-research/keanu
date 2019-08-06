package io.improbable.keanu.tensor.lng;

import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.jvm.JVMTensor;

import java.util.Arrays;
import java.util.stream.Collectors;


public class JVMLongTensorFactory implements LongTensorFactory {

    @Override
    public LongTensor create(int[] values, long[] shape) {
        return create(toLongs(values), shape);
    }

    @Override
    public LongTensor create(long[] values) {
        return create(values, values.length);
    }

    @Override
    public LongTensor create(int[] values) {
        return create(toLongs(values), values.length);
    }

    @Override
    public JVMLongTensor create(long[] values, long... shape) {
        if (values.length != TensorShape.getLength(shape)) {
            throw new IllegalArgumentException("Shape " + Arrays.toString(shape) + " does not match buffer size " + values.length);
        }
        return new JVMLongTensor(values, shape);
    }

    @Override
    public JVMLongTensor create(long value, long... shape) {
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

    @Override
    public LongTensor create(int value, long[] shape) {
        return create((long) value, shape);
    }

    @Override
    public LongTensor eye(long n) {
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

    private long[] toLongs(int[] ints) {
        final long[] lngs = new long[ints.length];
        for (int i = 0; i < ints.length; i++) {
            lngs[i] = ints[i];
        }
        return lngs;
    }

    public JVMLongTensor scalar(long scalarValue) {
        return new JVMLongTensor(scalarValue);
    }

    @Override
    public LongTensor scalar(int scalarValue) {
        return scalar((long) scalarValue);
    }

    @Override
    public JVMLongTensor ones(long... shape) {
        return create(1L, shape);
    }

    @Override
    public JVMLongTensor zeros(long... shape) {
        return create(0L, shape);
    }

    @Override
    public JVMLongTensor arange(Long start, Long end) {
        return arange(start, end, 1L);
    }

    @Override
    public JVMLongTensor arange(Long start, Long end, Long stepSize) {
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

    @Override
    public JVMLongTensor concat(int dimension, LongTensor... toConcat) {
        return new JVMLongTensor(
            JVMTensor.concat(JVMLongTensor.factory, toConcat, dimension,
                Arrays.stream(toConcat)
                    .map(tensor -> JVMLongTensor.asJVM(tensor).getBuffer())
                    .collect(Collectors.toList())
            ));
    }
}
