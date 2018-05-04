package io.improbable.keanu.vertices.dbltensor;

import java.util.Arrays;

public class TensorShape {

    private int[] shape;

    public TensorShape(int[] shape) {
        this.shape = shape;
    }

    public int[] getShape() {
        return shape;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        TensorShape that = (TensorShape) o;

        return Arrays.equals(shape, that.shape);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(shape);
    }
}
