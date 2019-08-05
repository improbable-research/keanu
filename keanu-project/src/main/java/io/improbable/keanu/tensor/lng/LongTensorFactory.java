package io.improbable.keanu.tensor.lng;

public interface LongTensorFactory {

    LongTensor create(long value, long[] shape);

    LongTensor create(long[] values, long[] shape);

    LongTensor create(long[] values);

    LongTensor ones(long[] shape);

    LongTensor zeros(long[] shape);

    LongTensor eye(long n);

    LongTensor arange(long start, long end);

    LongTensor arange(long start, long end, long stepSize);

    LongTensor scalar(long scalarValue);

    LongTensor concat(int dimension, LongTensor... toConcat);
}
