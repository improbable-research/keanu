package io.improbable.keanu.vertices.dbltensor;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

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
        return new Nd4jDoubleTensor(tensor.rdiv(1.0));
    }

    @Override
    public DoubleTensor minus(double value) {
        return new Nd4jDoubleTensor(tensor.sub(value));
    }

    @Override
    public DoubleTensor plus(double value) {
        return new Nd4jDoubleTensor(tensor.add(value));
    }

    @Override
    public DoubleTensor times(double value) {
        return new Nd4jDoubleTensor(tensor.mul(value));
    }

    @Override
    public DoubleTensor div(double value) {
        return new Nd4jDoubleTensor(tensor.div(value));
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
        return new Nd4jDoubleTensor(Transforms.log(tensor));
    }

    @Override
    public DoubleTensor sin() {
        return new Nd4jDoubleTensor(Transforms.sin(tensor));
    }

    @Override
    public DoubleTensor cos() {
        return new Nd4jDoubleTensor(Transforms.cos(tensor));
    }

    @Override
    public DoubleTensor asin() {
        return new Nd4jDoubleTensor(Transforms.asin(tensor));
    }

    @Override
    public DoubleTensor acos() {
        return new Nd4jDoubleTensor(Transforms.acos(tensor));
    }

    @Override
    public DoubleTensor exp() {
        return new Nd4jDoubleTensor(Transforms.exp(tensor));
    }

    @Override
    public DoubleTensor minus(DoubleTensor that) {
        return new Nd4jDoubleTensor(tensor.sub(((Nd4jDoubleTensor) that).tensor));
    }

    @Override
    public DoubleTensor plus(DoubleTensor that) {
        return new Nd4jDoubleTensor(tensor.add(((Nd4jDoubleTensor) that).tensor));
    }

    @Override
    public DoubleTensor times(DoubleTensor that) {
        return new Nd4jDoubleTensor(tensor.mul(((Nd4jDoubleTensor) that).tensor));
    }

    @Override
    public DoubleTensor div(DoubleTensor that) {
        return new Nd4jDoubleTensor(tensor.div(((Nd4jDoubleTensor) that).tensor));
    }

    @Override
    public DoubleTensor unaryMinus() {
        return new Nd4jDoubleTensor(tensor.neg());
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
