package io.improbable.keanu.vertices.dbltensor;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Nd4jDoubleTensor implements DoubleTensor {

    private INDArray tensor;

    public Nd4jDoubleTensor(double[] data, int[] shape) {
        tensor = Nd4j.create(data, shape);
    }

    public Nd4jDoubleTensor(INDArray tensor) {
        this.tensor = tensor;
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

    public double sum() {
        return tensor.sumNumber().doubleValue();
    }

    @Override
    public double scalar() {
        return tensor.getDouble(0);
    }

    @Override
    public DoubleTensor reciprocal() {
        return null;
    }

    @Override
    public DoubleTensor minus(double value) {
        return null;
    }

    @Override
    public DoubleTensor plus(double value) {
        return null;
    }

    @Override
    public DoubleTensor times(double value) {
        return null;
    }

    @Override
    public DoubleTensor div(double value) {
        return null;
    }

    @Override
    public DoubleTensor pow(DoubleTensor exponent) {
        return null;
    }

    @Override
    public DoubleTensor pow(double exponent) {
        return null;
    }

    @Override
    public DoubleTensor log() {
        return null;
    }

    @Override
    public DoubleTensor sin() {
        return null;
    }

    @Override
    public DoubleTensor cos() {
        return null;
    }

    @Override
    public DoubleTensor asin() {
        return null;
    }

    @Override
    public DoubleTensor acos() {
        return null;
    }

    @Override
    public DoubleTensor exp() {
        return null;
    }

    @Override
    public DoubleTensor minus(DoubleTensor that) {
        return null;
    }

    @Override
    public DoubleTensor plus(DoubleTensor that) {
        return null;
    }

    @Override
    public DoubleTensor times(DoubleTensor that) {
        return null;
    }

    @Override
    public DoubleTensor div(DoubleTensor that) {
        return null;
    }

    @Override
    public DoubleTensor unaryMinus() {
        return null;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        Nd4jDoubleTensor that = (Nd4jDoubleTensor) o;

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
}
