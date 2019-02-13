package io.improbable.keanu.tensor.dbl;

public interface DoubleTensorFactory {

    DoubleTensor create(double value, long[] shape);

    DoubleTensor create(double[] values, long[] shape);

    DoubleTensor create(double[] values);

    DoubleTensor ones(long[] shape);

    DoubleTensor zeros(long[] shape);

    DoubleTensor eye(long n);

    DoubleTensor linspace(double start, double end, int numberOfPoints);

    DoubleTensor arange(double start, double end);

    DoubleTensor arange(double start, double end, double stepSize);

    DoubleTensor scalar(double scalarValue);

    DoubleTensor concat(int dimension, DoubleTensor... toConcat);
}
