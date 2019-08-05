package io.improbable.keanu.tensor.lng;

public class JVMLongTensorFactory implements LongTensorFactory {

    @Override
    public LongTensor create(Long value, long[] shape) {
        return JVMLongTensor.create(value, shape);
    }

    @Override
    public LongTensor create(long[] values, long[] shape) {
        return JVMLongTensor.create(values, shape);
    }

    @Override
    public LongTensor create(int[] values, long[] shape) {
        return create(toLongs(values), shape);
    }

    @Override
    public LongTensor create(long[] values) {
        return JVMLongTensor.create(values, values.length);
    }

    @Override
    public LongTensor create(int[] values) {
        return create(toLongs(values));
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
    public LongTensor arange(Long start, Long end) {
        return JVMLongTensor.arange(start, end);
    }

    @Override
    public LongTensor arange(Long start, Long end, Long stepSize) {
        return JVMLongTensor.arange(start, end, stepSize);
    }

    @Override
    public LongTensor scalar(Long scalarValue) {
        return JVMLongTensor.scalar(scalarValue);
    }

    @Override
    public LongTensor concat(int dimension, LongTensor... toConcat) {
        return JVMLongTensor.concat(dimension, toConcat);
    }

    private long[] toLongs(int[] ints) {
        final long[] lngs = new long[ints.length];
        for (int i = 0; i < ints.length; i++) {
            lngs[i] = ints[i];
        }
        return lngs;
    }
}
