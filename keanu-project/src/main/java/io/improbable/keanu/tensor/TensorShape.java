package io.improbable.keanu.tensor;

import java.util.Arrays;

public class TensorShape {

    private long[] shape;

    public TensorShape(long[] shape) {
        this.shape = Arrays.copyOf(shape, shape.length);
    }

    public long[] getShape() {
        return Arrays.copyOf(shape, shape.length);
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
    public static long getLength(long[] shape) {
        if (shape.length == 0) {
            return 0;
        } else {
            long length = 1;
            for (long dim : shape) {
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
    public static long[] getRowFirstStride(long[] shape) {
        long[] stride = new long[shape.length];
        stride[stride.length - 1] = 1;

        int buffer = 1;
        for (int i = stride.length - 2; i >= 0; i--) {
            buffer *= shape[i + 1];
            stride[i] = buffer;
        }

        return stride;
    }

    /**
     * @param shape  shape to find the index for
     * @param stride stride to find the index for
     * @param index  the index in each dimension
     * @return the flat index from a N dimensional index
     */
    public static long getFlatIndex(long[] shape, long[] stride, long... index) {
        long flatIndex = 0;
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

    /**
     * This method can be interpreted as the opposite to getFlatIndex.
     *
     * @param shape     the shape to find the index of
     * @param stride    the stride to find the index of
     * @param flatIndex the index to f
     * @return converts from a flat index to a N dimensional index. Where N = the dimensionality of the shape.
     */
    public static long[] getShapeIndices(long[] shape, long[] stride, long flatIndex) {
        if (flatIndex > getLength(shape)) {
            throw new IllegalArgumentException("The requested index is out of the bounds of this shape.");
        }
        long[] shapeIndices = new long[stride.length];
        long remainder = flatIndex;
        for (int i = 0; i < stride.length; i++) {
            shapeIndices[i] = remainder / stride[i];
            remainder -= shapeIndices[i] * stride[i];
        }
        return shapeIndices;
    }

    public static boolean isScalar(long[] shape) {
        return getLength(shape) == 1;
    }

    public static long[] concat(long[] shape1, long[] shape2) {
        long[] result = new long[shape1.length + shape2.length];
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

    public static long[] selectDimensions(int from, int to, long[] shape) {
        if (from > to) {
            throw new IllegalArgumentException("to dimension must be less than from");
        }

        long[] newShape = new long[to - from];

        for (int i = 0; i < (to - from); i++) {
            newShape[i] = shape[i + from];
        }

        return newShape;
    }
    
    public static long[] shapeDesiredToRankByAppendingOnes(long[] lowRankTensorShape, int desiredRank) {
        return increaseRankByPaddingValue(lowRankTensorShape, desiredRank, true, 1);
    }

    public static long[] shapeToDesiredRankByPrependingOnes(long[] lowRankTensorShape, int desiredRank) {
        return increaseRankByPaddingValue(lowRankTensorShape, desiredRank, false, 1);
    }

    public static long[] shapeToDesiredRankByPrependingNegOnes(long[] lowRankTensorShape, int desiredRank) {
        return increaseRankByPaddingValue(lowRankTensorShape, desiredRank, false, -1);
    }

    private static long[] increaseRankByPaddingValue(long[] lowRankTensorShape, int desiredRank, boolean append, int val) {
        long[] paddedShape = new long[desiredRank];
        if (lowRankTensorShape.length > desiredRank) {
            throw new IllegalArgumentException("low rank tensor must be rank less than or equal to desired rank");
        }

        Arrays.fill(paddedShape, val);
        if (append) {
            System.arraycopy(lowRankTensorShape, 0, paddedShape, 0, lowRankTensorShape.length);
        } else {
            System.arraycopy(lowRankTensorShape, 0, paddedShape, paddedShape.length - lowRankTensorShape.length, lowRankTensorShape.length);
        }

        return paddedShape;
    }

    public static long[] shapeSlice(int dimension, long[] shape) {
        long[] newShape = Arrays.copyOf(shape, shape.length);
        newShape[dimension] = 1;
        return newShape;
    }

}

