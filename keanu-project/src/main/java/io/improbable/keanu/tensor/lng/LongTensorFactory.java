package io.improbable.keanu.tensor.lng;

import io.improbable.keanu.tensor.FixedPointTensorFactory;

public interface LongTensorFactory extends FixedPointTensorFactory<Long, LongTensor> {

    @Override
    LongTensor create(Long value, long[] shape);

    @Override
    LongTensor create(long[] values, long[] shape);

    @Override
    LongTensor create(int[] values, long[] shape);

    @Override
    LongTensor create(long[] values);

    @Override
    LongTensor create(int[] values);

    @Override
    LongTensor ones(long[] shape);

    @Override
    LongTensor zeros(long[] shape);

    @Override
    LongTensor eye(long n);

    @Override
    LongTensor arange(Long start, Long end);

    @Override
    LongTensor arange(Long start, Long end, Long stepSize);

    @Override
    LongTensor scalar(Long scalarValue);

    LongTensor concat(int dimension, LongTensor... toConcat);
}
