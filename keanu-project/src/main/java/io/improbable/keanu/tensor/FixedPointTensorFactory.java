package io.improbable.keanu.tensor;

public interface FixedPointTensorFactory<N extends Number, TENSOR extends FixedPointTensor<N, TENSOR>> {

    TENSOR create(N value, long[] shape);

    TENSOR create(int[] values, long[] shape);

    TENSOR create(long[] values, long[] shape);

    TENSOR create(int... values);

    TENSOR create(long... values);

    TENSOR ones(long[] shape);

    TENSOR zeros(long[] shape);

    TENSOR eye(long n);

    TENSOR arange(N start, N end);

    TENSOR arange(N start, N end, N stepSize);

    TENSOR scalar(N scalarValue);

}
