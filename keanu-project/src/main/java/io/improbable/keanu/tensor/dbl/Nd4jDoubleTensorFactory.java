package io.improbable.keanu.tensor.dbl;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

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

    @Override
    public DoubleTensor concat(int dimension, DoubleTensor... toConcat) {
        INDArray[] concatAsINDArray = new INDArray[toConcat.length];
        for (int i = 0; i < toConcat.length; i++) {
            concatAsINDArray[i] = Nd4jDoubleTensor.unsafeGetNd4J(toConcat[i]).dup();
            if (concatAsINDArray[i].shape().length == 0) {
                concatAsINDArray[i] = concatAsINDArray[i].reshape(1);
            }
        }
        INDArray concat = Nd4j.concat(dimension, concatAsINDArray);
        return new Nd4jDoubleTensor(concat);
    }
}
