package io.improbable.keanu.vertices.dbltensor;

public class SimpleScalarTensor implements DoubleTensor {

    private double scalar;

    public SimpleScalarTensor(double scalar) {
        this.scalar = scalar;
    }

    @Override
    public double scalar() {
        return scalar;
    }

    @Override
    public int getLength() {
        return 1;
    }
}
