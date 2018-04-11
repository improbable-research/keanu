package io.improbable.keanu.vertices.dbl.tensor;


import java.util.Arrays;

public interface Tensor {

    int getRank();

    int[] getShape();

    int getLength();

    default boolean isScalar() {
        return getRank() == 0;
    }

    default boolean isVector() {
        return getRank() == 1;
    }

    default boolean isMatrix() {
        return getRank() == 2;
    }

    default boolean hasSameShapeAs(Tensor that) {
        return Arrays.equals(this.getShape(), that.getShape());
    }
}
