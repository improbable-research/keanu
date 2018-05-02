package io.improbable.keanu.vertices.dbltensor;

public class SimpleScalarTensor implements DoubleTensor {

    private double scalar;

    public SimpleScalarTensor(double scalar) {
        this.scalar = scalar;
    }

    @Override
    public double getValue(int[] index) {
        return scalar;
    }

    @Override
    public void setValue(double value, int[] index) {
        this.scalar = value;
    }

    @Override
    public double sum() {
        return scalar;
    }

    @Override
    public double scalar() {
        return scalar;
    }

    @Override
    public DoubleTensor reciprocal() {
        return new SimpleScalarTensor(1.0 / scalar);
    }

    @Override
    public DoubleTensor minus(double value) {
        return new SimpleScalarTensor(scalar - value);
    }

    @Override
    public DoubleTensor plus(double value) {
        return new SimpleScalarTensor(scalar + value);
    }

    @Override
    public DoubleTensor times(double value) {
        return new SimpleScalarTensor(scalar * value);
    }

    @Override
    public DoubleTensor div(double value) {
        return new SimpleScalarTensor(scalar / value);
    }

    @Override
    public DoubleTensor pow(DoubleTensor exponent) {
        if (exponent.isScalar()) {
            return new SimpleScalarTensor(Math.pow(scalar, exponent.scalar()));
        }

        throw new IllegalArgumentException("Only scalar tensors supported");
    }

    @Override
    public DoubleTensor pow(double exponent) {
        return new SimpleScalarTensor(Math.pow(scalar, exponent));
    }

    @Override
    public DoubleTensor log() {
        return new SimpleScalarTensor(Math.log(scalar));
    }

    @Override
    public DoubleTensor sin() {
        return new SimpleScalarTensor(Math.sin(scalar));
    }

    @Override
    public DoubleTensor cos() {
        return new SimpleScalarTensor(Math.cos(scalar));
    }

    @Override
    public DoubleTensor asin() {
        return new SimpleScalarTensor(Math.asin(scalar));
    }

    @Override
    public DoubleTensor acos() {
        return new SimpleScalarTensor(Math.acos(scalar));
    }

    @Override
    public DoubleTensor exp() {
        return new SimpleScalarTensor(Math.exp(scalar));
    }

    @Override
    public DoubleTensor minus(DoubleTensor that) {
        if (that.isScalar()) {
            return new SimpleScalarTensor(scalar - that.scalar());
        }
        throw new IllegalArgumentException("Only scalar tensors supported");
    }

    @Override
    public DoubleTensor plus(DoubleTensor that) {
        if (that.isScalar()) {
            return new SimpleScalarTensor(scalar + that.scalar());
        }
        throw new IllegalArgumentException("Only scalar tensors supported");
    }

    @Override
    public DoubleTensor times(DoubleTensor that) {
        if (that.isScalar()) {
            return new SimpleScalarTensor(scalar * that.scalar());
        }
        throw new IllegalArgumentException("Only scalar tensors supported");
    }

    @Override
    public DoubleTensor div(DoubleTensor that) {
        if (that.isScalar()) {
            return new SimpleScalarTensor(scalar / that.scalar());
        }
        throw new IllegalArgumentException("Only scalar tensors supported");
    }

    @Override
    public DoubleTensor unaryMinus() {
        return new SimpleScalarTensor(-scalar);
    }

    @Override
    public int getRank() {
        return 0;
    }

    @Override
    public int[] getShape() {
        return new int[0];
    }

    @Override
    public int getLength() {
        return 1;
    }
}
