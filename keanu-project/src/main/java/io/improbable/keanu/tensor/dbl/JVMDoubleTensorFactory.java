package io.improbable.keanu.tensor.dbl;

public class JVMDoubleTensorFactory implements DoubleTensorFactory {

    @Override
    public DoubleTensor create(double value, long[] shape) {
        return JVMDoubleTensor.create(value, shape);
    }

    @Override
    public DoubleTensor create(double[] values, long[] shape) {
        return JVMDoubleTensor.create(values, shape);
    }

    @Override
    public DoubleTensor create(double[] values) {
        return JVMDoubleTensor.create(values);
    }

    @Override
    public DoubleTensor ones(long[] shape) {
        return JVMDoubleTensor.ones(shape);
    }

    @Override
    public DoubleTensor zeros(long[] shape) {
        return JVMDoubleTensor.zeros(shape);
    }

    @Override
    public DoubleTensor eye(long n) {
        return JVMDoubleTensor.eye(n);
    }

    @Override
    public DoubleTensor linspace(double start, double end, int numberOfPoints) {
        return null;
    }

    @Override
    public DoubleTensor arange(double start, double end) {
        return null;
    }

    @Override
    public DoubleTensor arange(double start, double end, double stepSize) {
        return null;
    }

    @Override
    public DoubleTensor scalar(double scalarValue) {
        return JVMDoubleTensor.scalar(scalarValue);
    }

    @Override
    public DoubleTensor concat(int dimension, DoubleTensor... toConcat) {
        return null;
    }
}
