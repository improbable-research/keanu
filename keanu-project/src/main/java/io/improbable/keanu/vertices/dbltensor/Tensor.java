package io.improbable.keanu.vertices.dbltensor;


import java.util.Arrays;

public interface Tensor {

    int getRank();

    int[] getShape();

    int getLength();

    default boolean isScalar() {
        return getLength() == 1;
    }

    default boolean isVector() {
        return getRank() == 1;
    }

    default boolean isMatrix() {
        return getRank() == 2;
    }

    default boolean hasSameShapeAs(Tensor that) {
        return hasSameShapeAs(that.getShape());
    }

    default boolean hasSameShapeAs(int[] shape) {
        return Arrays.equals(this.getShape(), shape);
    }

}
