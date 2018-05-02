package io.improbable.keanu.vertices.dbltensor;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class DoubleTensor implements Tensor {

    private INDArray tensor;

    public DoubleTensor(double[] data, int[] shape) {
        tensor = Nd4j.create(data, shape);
    }

    @Override
    public int getRank() {
        return tensor.rank();
    }

    @Override
    public int[] getShape() {
        return tensor.shape();
    }

    @Override
    public int getLength() {
        return tensor.length();
    }

    public double getValue(int[] index) {
        return tensor.getDouble(index);
    }

    public void setValue(double value, int[] index) {
        tensor.putScalar(index, value);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        DoubleTensor that = (DoubleTensor) o;

        return tensor.equals(that.tensor);
    }

    @Override
    public int hashCode() {
        return tensor.hashCode();
    }

    @Override
    public String toString() {
        return tensor.toString();
    }

    public double sum() {
        return tensor.sumNumber().doubleValue();
    }
}
