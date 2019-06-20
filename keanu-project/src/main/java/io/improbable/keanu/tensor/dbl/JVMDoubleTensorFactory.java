package io.improbable.keanu.tensor.dbl;

import io.improbable.keanu.tensor.TensorShape;

public class JVMDoubleTensorFactory implements DoubleTensorFactory {

    @Override
    public DoubleTensor create(double value, long[] shape) {
        if (TensorShape.getLengthAsInt(shape) == 1) {
            return new ScalarDoubleTensor(value, shape);
        }
        return JVMDoubleTensor.create(value, shape);
    }

    @Override
    public DoubleTensor create(double[] values, long[] shape) {
        if (TensorShape.getLengthAsInt(shape) == 1) {
            return new ScalarDoubleTensor(values[0], shape);
        }
        return JVMDoubleTensor.create(values, shape);
    }

    @Override
    public DoubleTensor create(double[] values) {
        return JVMDoubleTensor.create(values);
    }

    @Override
    public DoubleTensor ones(long[] shape) {
        if (TensorShape.getLengthAsInt(shape) == 1) {
            return new ScalarDoubleTensor(1.0, shape);
        }
        return JVMDoubleTensor.ones(shape);
    }

    @Override
    public DoubleTensor zeros(long[] shape) {
        if (TensorShape.getLengthAsInt(shape) == 1) {
            return new ScalarDoubleTensor(0.0, shape);
        }
        return JVMDoubleTensor.zeros(shape);
    }

    @Override
    public DoubleTensor eye(long n) {
        if (n == 0) {
            return new ScalarDoubleTensor(1.0, new long[]{1, 1});
        }
        return JVMDoubleTensor.eye(n);
    }

    @Override
    public DoubleTensor linspace(double start, double end, int numberOfPoints) {
        return JVMDoubleTensor.linspace(start, end, numberOfPoints);
    }

    @Override
    public DoubleTensor arange(double start, double end) {
        return JVMDoubleTensor.arange(start, end);
    }

    @Override
    public DoubleTensor arange(double start, double end, double stepSize) {
        return JVMDoubleTensor.arange(start, end, stepSize);
    }

    @Override
    public DoubleTensor scalar(double scalarValue) {
        return new ScalarDoubleTensor(scalarValue);
    }

    @Override
    public DoubleTensor concat(int dimension, DoubleTensor... toConcat) {
        return JVMDoubleTensor.concat(dimension, toConcat);
    }
}
