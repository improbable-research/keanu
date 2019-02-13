package io.improbable.keanu.tensor.dbl;

public class Nd4jDoubleTensorFactory implements DoubleTensorFactory {

    @Override
    public DoubleTensor create(double value, long[] shape) {
        return Nd4jDoubleTensor.create(value, shape);
    }

    @Override
    public DoubleTensor create(double[] values, long[] shape) {
        return Nd4jDoubleTensor.create(values, shape);
    }

    @Override
    public DoubleTensor create(double[] values) {
        return Nd4jDoubleTensor.create(values);
    }

    @Override
    public DoubleTensor ones(long[] shape) {
        return Nd4jDoubleTensor.ones(shape);
    }

    @Override
    public DoubleTensor zeros(long[] shape) {
        return Nd4jDoubleTensor.zeros(shape);
    }

    @Override
    public DoubleTensor eye(long n) {
        return Nd4jDoubleTensor.eye(n);
    }

    @Override
    public DoubleTensor linspace(double start, double end, int numberOfPoints) {
        return Nd4jDoubleTensor.linspace(start, end, numberOfPoints);
    }

    @Override
    public DoubleTensor arange(double start, double end) {
        return Nd4jDoubleTensor.arange(start, end);
    }

    @Override
    public DoubleTensor arange(double start, double end, double stepSize) {
        return Nd4jDoubleTensor.arange(start, end, stepSize);
    }

    @Override
    public DoubleTensor scalar(double scalarValue) {
        return Nd4jDoubleTensor.scalar(scalarValue);
    }
}
