package io.improbable.keanu.tensor;

import io.improbable.keanu.tensor.dbl.DoubleTensor;

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
        return getLength(shape) == 1;
    }

    public static int[] concat(int[] shape1, int[] shape2) {
        int[] result = new int[shape1.length + shape2.length];
        System.arraycopy(shape1, 0, result, 0, shape1.length);
        System.arraycopy(shape2, 0, result, shape1.length, shape2.length);
        return result;
    }

    /**
     * @param fromDimension starting from and including this dimension
     * @param toDimension   up to but excluding this dimension
     * @return an int array containing the dimension numbers from a given dimension to a higher
     * dimension. e.g. dimensionRange(0, 3) = int[]{0, 1, 2}
     */
    public static int[] dimensionRange(int fromDimension, int toDimension) {
        if (fromDimension > toDimension) {
            throw new IllegalArgumentException("from dimension must be less than to dimension");
        }

        int dimensionCount = toDimension - fromDimension;
        int[] dims = new int[dimensionCount];
        for (int i = 0; i < dimensionCount; i++) {
            dims[i] = i + fromDimension;
        }
        return dims;
    }
}
