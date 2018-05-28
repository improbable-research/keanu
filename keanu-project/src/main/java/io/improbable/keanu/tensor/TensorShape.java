package io.improbable.keanu.tensor;

import java.util.Arrays;

public class TensorShape {

    private int[] shape;

    public TensorShape(int[] shape) {
        this.shape = shape;
    }

    public int[] getShape() {
        return shape;
    }

    public boolean isScalar() {
        return isScalar(shape);
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

    /**
     * @param shape for finding length of
     * @return the number of elements in a tensor. This
     * is the product of all ints in shape.
     */
    public static long getLength(int[] shape) {
        if (shape.length == 0) {
            return 0;
        } else {
            long length = 1;
            for (int dim : shape) {
                length *= dim;
            }
            return length;
        }
    }

    /**
     * @param shape shape to find stride for
     * @return the stride which is used to convert from a N dimensional index
     * to a buffer array flat index. This is based on the C convention of
     * row first instead of column.
     */
    public static int[] getRowFirstStride(int[] shape) {
        int[] stride = new int[shape.length];
        stride[stride.length - 1] = 1;

        int buffer = 1;
        for (int i = stride.length - 2; i >= 0; i--) {
            buffer *= shape[i + 1];
            stride[i] = buffer;
        }

        return stride;
    }

    public static int getFlatIndex(int[] shape, int[] stride, int... index) {
        int flatIndex = 0;
        for (int i = 0; i < index.length; i++) {

            if (index[i] >= shape[i]) {
                throw new IllegalArgumentException(
                    "Invalid index " + Arrays.toString(index) + " for shape " + Arrays.toString(shape)
                );
            }

            flatIndex += stride[i] * index[i];
        }
        return flatIndex;
    }

    public static boolean isScalar(int[] shape) {
        return Arrays.equals(Tensor.SCALAR_SHAPE, shape);
    }
}
