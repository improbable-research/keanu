package io.improbable.keanu.tensor.lng;

import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.jvm.JVMFixedPointTensor;
import io.improbable.keanu.tensor.jvm.ResultWrapper;

import java.util.Arrays;

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
    protected JVMLongTensor create(LongBuffer.PrimitiveLongWrapper buffer, long[] shape, long[] stride) {
        return new JVMLongTensor(buffer, shape, stride);
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
        return null;
    }

}
