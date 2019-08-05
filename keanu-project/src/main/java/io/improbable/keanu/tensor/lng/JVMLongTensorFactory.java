package io.improbable.keanu.tensor.lng;

public class JVMLongTensorFactory implements LongTensorFactory {

    @Override
    public LongTensor create(long value, long[] shape) {
        return JVMLongTensor.create(value, shape);
    }

    @Override
    public LongTensor create(long[] values, long[] shape) {
        return JVMLongTensor.create(values, shape);
    }

    @Override
    public LongTensor create(long[] values) {
        return JVMLongTensor.create(values);
    }

    @Override
    public LongTensor ones(long[] shape) {
        return JVMLongTensor.ones(shape);
    }

    @Override
    public LongTensor zeros(long[] shape) {
        return JVMLongTensor.zeros(shape);
    }

    @Override
    public LongTensor eye(long n) {
        return JVMLongTensor.eye(n);
    }

    @Override
    public LongTensor arange(long start, long end) {
        return JVMLongTensor.arange(start, end);
    }

    @Override
    public LongTensor arange(long start, long end, long stepSize) {
        return JVMLongTensor.arange(start, end, stepSize);
    }

    @Override
    public LongTensor scalar(long scalarValue) {
        return JVMLongTensor.scalar(scalarValue);
    }

    @Override
    public LongTensor concat(int dimension, LongTensor... toConcat) {
        return JVMLongTensor.concat(dimension, toConcat);
    }
}
