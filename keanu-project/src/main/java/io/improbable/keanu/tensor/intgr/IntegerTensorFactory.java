package io.improbable.keanu.tensor.intgr;

import io.improbable.keanu.tensor.FixedPointTensorFactory;

public interface IntegerTensorFactory extends FixedPointTensorFactory<Integer, IntegerTensor> {

    @Override
    IntegerTensor create(Integer value, long[] shape);

    @Override
    IntegerTensor create(long[] values, long[] shape);

    @Override
    IntegerTensor create(int[] values, long[] shape);

    @Override
    IntegerTensor create(long[] values);

    @Override
    IntegerTensor create(int[] values);

    @Override
    IntegerTensor ones(long[] shape);

    @Override
    IntegerTensor zeros(long[] shape);

    @Override
    IntegerTensor eye(long n);

    @Override
    IntegerTensor arange(Integer start, Integer end);

    @Override
    IntegerTensor arange(Integer start, Integer end, Integer stepSize);

    @Override
    IntegerTensor scalar(Integer scalarValue);

    IntegerTensor concat(int dimension, IntegerTensor... toConcat);
}
