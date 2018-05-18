package io.improbable.keanu.vertices.dbltensor;

import java.util.function.Function;

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
    public DoubleTensor apply(Function<Double, Double> f) {
        return new SimpleScalarTensor(f.apply(scalar));
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
    public DoubleTensor sqrt() {
        return pow(0.5);
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
    public DoubleTensor abs() {
        return new SimpleScalarTensor(Math.abs(scalar));
    }

    @Override
    public DoubleTensor unaryMinus() {
        return new SimpleScalarTensor(-scalar);
    }

    @Override
    public DoubleTensor getGreaterThanMask(DoubleTensor greaterThanThis) {
        if (greaterThanThis.isScalar()) {
            return new SimpleScalarTensor(scalar > greaterThanThis.scalar() ? 1 : 0);
        }
        throw new IllegalArgumentException("Only scalar tensors supported");
    }

    @Override
    public DoubleTensor getLessThanMask(DoubleTensor lessThanThis) {
        if (lessThanThis.isScalar()) {
            return new SimpleScalarTensor(scalar < lessThanThis.scalar() ? 1 : 0);
        }
        throw new IllegalArgumentException("Only scalar tensors supported");
    }

    @Override
    public DoubleTensor getLessThanOrEqualToMask(DoubleTensor lessThanOrEqualToThis) {
        if (lessThanOrEqualToThis.isScalar()) {
            return new SimpleScalarTensor(scalar <= lessThanOrEqualToThis.scalar() ? 1 : 0);
        }
        throw new IllegalArgumentException("Only scalar tensors supported");
    }

    @Override
    public DoubleTensor applyWhere(DoubleTensor withMask, double value) {
        if (withMask.isScalar()) {
            return new SimpleScalarTensor(withMask.scalar() == 1.0 ? value : scalar);
        }
        throw new IllegalArgumentException("Only scalar tensors supported");
    }

    @Override
    public DoubleTensor reciprocalInPlace() {
        return reciprocal();
    }

    @Override
    public DoubleTensor minusInPlace(double value) {
        return minusInPlace(value);
    }

    @Override
    public DoubleTensor plusInPlace(double value) {
        return plus(value);
    }

    @Override
    public DoubleTensor timesInPlace(double value) {
        return times(value);
    }

    @Override
    public DoubleTensor divInPlace(double value) {
        return div(value);
    }

    @Override
    public DoubleTensor powInPlace(DoubleTensor exponent) {
        return pow(exponent);
    }

    @Override
    public DoubleTensor powInPlace(double exponent) {
        return pow(exponent);
    }

    @Override
    public DoubleTensor sqrtInPlace() {
        return pow(0.5);
    }

    @Override
    public DoubleTensor logInPlace() {
        return log();
    }

    @Override
    public DoubleTensor sinInPlace() {
        return sin();
    }

    @Override
    public DoubleTensor cosInPlace() {
        return cos();
    }

    @Override
    public DoubleTensor asinInPlace() {
        return asin();
    }

    @Override
    public DoubleTensor acosInPlace() {
        return acos();
    }

    @Override
    public DoubleTensor expInPlace() {
        return exp();
    }

    @Override
    public DoubleTensor minusInPlace(DoubleTensor that) {
        return minus(that);
    }

    @Override
    public DoubleTensor plusInPlace(DoubleTensor that) {
        return plus(that);
    }

    @Override
    public DoubleTensor timesInPlace(DoubleTensor that) {
        return times(that);
    }

    @Override
    public DoubleTensor divInPlace(DoubleTensor that) {
        return div(that);
    }

    @Override
    public DoubleTensor unaryMinusInPlace() {
        return unaryMinus();
    }

    @Override
    public DoubleTensor absInPlace() {
        return abs();
    }

    @Override
    public DoubleTensor applyInPlace(Function<Double, Double> f) {
        return apply(f);
    }

    @Override
    public double[] getLinearView() {
        return new double[]{scalar};
    }

    @Override
    public int getRank() {
        return 2;
    }

    @Override
    public int[] getShape() {
        return new int[]{1, 1};
    }

    @Override
    public int getLength() {
        return 1;
    }

}
