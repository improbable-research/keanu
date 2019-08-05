package io.improbable.keanu.tensor.intgr;

import com.google.common.primitives.Ints;

public class Nd4jIntegerTensorFactory implements IntegerTensorFactory {

    @Override
    public IntegerTensor create(long value, long[] shape) {
        return Nd4jIntegerTensor.create(Ints.checkedCast(value), shape);
    }

    @Override
    public IntegerTensor create(int value, long[] shape) {
        return Nd4jIntegerTensor.create(value, shape);
    }

    @Override
    public IntegerTensor create(int[] values, long[] shape) {
        return Nd4jIntegerTensor.create(values, shape);
    }

    @Override
    public IntegerTensor create(long[] values, long[] shape) {
        return create(toIntegers(values), shape);
    }

    @Override
    public IntegerTensor create(int[] values) {
        return Nd4jIntegerTensor.create(values, values.length);
    }

    @Override
    public IntegerTensor create(long[] values) {
        return create(toIntegers(values));
    }

    @Override
    public IntegerTensor ones(long[] shape) {
        return Nd4jIntegerTensor.ones(shape);
    }

    @Override
    public IntegerTensor zeros(long[] shape) {
        return Nd4jIntegerTensor.zeros(shape);
    }

    @Override
    public IntegerTensor eye(long n) {
        return Nd4jIntegerTensor.eye(n);
    }

    @Override
    public IntegerTensor arange(Integer start, Integer end) {
        return Nd4jIntegerTensor.arange(start, end);
    }

    @Override
    public IntegerTensor arange(Integer start, Integer end, Integer stepSize) {
        return Nd4jIntegerTensor.arange(start, end, stepSize);
    }

    @Override
    public IntegerTensor scalar(long scalarValue) {
        return Nd4jIntegerTensor.scalar(Ints.checkedCast(scalarValue));
    }

    @Override
    public IntegerTensor scalar(int scalarValue) {
        return Nd4jIntegerTensor.scalar(scalarValue);
    }

    @Override
    public IntegerTensor concat(int dimension, IntegerTensor... toConcat) {
        return IntegerTensor.concat(dimension, toConcat);
    }

    private int[] toIntegers(long[] lngs) {
        final int[] ints = new int[lngs.length];
        for (int i = 0; i < ints.length; i++) {
            ints[i] = Ints.checkedCast(lngs[i]);
        }
        return ints;
    }
}
