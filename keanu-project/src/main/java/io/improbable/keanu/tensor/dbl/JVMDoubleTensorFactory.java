package io.improbable.keanu.tensor.dbl;

import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.jvm.JVMTensor;

import java.util.Arrays;
import java.util.stream.Collectors;

public class JVMDoubleTensorFactory implements DoubleTensorFactory {

    public static final JVMDoubleTensorFactory INSTANCE = new JVMDoubleTensorFactory();

    @Override
    public DoubleTensor create(double[] values) {
        return create(values, values.length);
    }

    @Override
    public JVMDoubleTensor create(double[] values, long... shape) {
        if (values.length != TensorShape.getLength(shape)) {
            throw new IllegalArgumentException("Shape " + Arrays.toString(shape) + " does not match buffer size " + values.length);
        }
        return new JVMDoubleTensor(values, shape);
    }

    @Override
    public JVMDoubleTensor create(double value, long... shape) {
        final int length = TensorShape.getLengthAsInt(shape);

        if (length > 1) {
            final double[] buffer = new double[length];
            if (value != 0) {
                Arrays.fill(buffer, value);
            }

            return new JVMDoubleTensor(buffer, shape);
        } else {
            return new JVMDoubleTensor(new DoubleBuffer.DoubleWrapper(value), shape);
        }
    }

    @Override
    public JVMDoubleTensor ones(long... shape) {
        return create(1.0, shape);
    }

    @Override
    public JVMDoubleTensor zeros(long... shape) {
        return create(0.0, shape);
    }

    @Override
    public JVMDoubleTensor eye(long n) {

        if (n == 1) {
            return create(1.0, 1, 1);
        } else {

            double[] buffer = new double[Ints.checkedCast(n * n)];
            int nInt = Ints.checkedCast(n);
            for (int i = 0; i < n; i++) {
                buffer[i * nInt + i] = 1;
            }
            return new JVMDoubleTensor(buffer, new long[]{n, n});
        }
    }

    @Override
    public JVMDoubleTensor linspace(double start, double end, int numberOfPoints) {
        Preconditions.checkArgument(numberOfPoints > 0);
        double stepSize = (end - start) / (numberOfPoints - 1);

        return linearBufferCreate(start, numberOfPoints, stepSize);
    }

    private static JVMDoubleTensor linearBufferCreate(double start, int numberOfPoints, double stepSize) {
        Preconditions.checkArgument(numberOfPoints > 0);
        double[] buffer = new double[numberOfPoints];

        double currentValue = start;
        for (int i = 0; i < buffer.length; i++, currentValue += stepSize) {
            buffer[i] = currentValue;
        }

        return new JVMDoubleTensor(buffer, new long[]{buffer.length});
    }

    @Override
    public JVMDoubleTensor arange(double start, double end) {
        return arange(start, end, 1.0);
    }

    @Override
    public JVMDoubleTensor arange(double start, double end, double stepSize) {
        Preconditions.checkArgument(stepSize != 0);
        int steps = (int) Math.ceil((end - start) / stepSize);

        return linearBufferCreate(start, steps, stepSize);
    }

    @Override
    public JVMDoubleTensor scalar(double scalarValue) {
        return new JVMDoubleTensor(scalarValue);
    }

    @Override
    public DoubleTensor concat(int dimension, DoubleTensor... toConcat) {
        return new JVMDoubleTensor(
            JVMTensor.concat(JVMDoubleTensor.factory, toConcat, dimension,
                Arrays.stream(toConcat)
                    .map(tensor -> JVMDoubleTensor.asJVM(tensor).getBuffer())
                    .collect(Collectors.toList())
            ));
    }
}
