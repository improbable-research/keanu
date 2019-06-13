package io.improbable.keanu.tensor;

import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import org.apache.commons.lang3.ArrayUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


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

    public boolean isLengthOne() {
        return isLengthOne(shape);
    }

    public int getRank() {
        return shape.length;
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
        long length = 1;
        for (long dim : shape) {
            length *= dim;
        }
        return length;
    }

    public static int getLengthAsInt(long[] shape) {
        return Ints.checkedCast(getLength(shape));
    }

    /**
     * @param shape shape to find stride for
     * @return the stride which is used to convert from a N dimensional index
     * to a buffer array flat index. This is based on the C convention of
     * row first instead of column.
     */
    public static long[] getRowFirstStride(long[] shape) {
        long[] stride = new long[shape.length];

        if (shape.length == 0) {
            return stride;
        }

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
        for (int i = 0; i < shape.length; i++) {

            if (i >= index.length || index[i] >= shape[i]) {
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
        return shape.length == 0;
    }

    public static boolean isLengthOne(long[] shape) {
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

    public static int[] slideDimension(int from, int to, int rank) {
        int[] dimensionRange = dimensionRange(0, rank);
        List<Integer> shapeList = new ArrayList<>(Ints.asList(dimensionRange));
        Integer dimLength = shapeList.remove(from);
        shapeList.add(to, dimLength);
        return Ints.toArray(shapeList);
    }

    public static long[] shapeDesiredToRankByAppendingOnes(long[] lowRankTensorShape, int desiredRank) {
        return increaseRankByPaddingValue(lowRankTensorShape, desiredRank, true);
    }

    public static long[] shapeToDesiredRankByPrependingOnes(long[] lowRankTensorShape, int desiredRank) {
        return increaseRankByPaddingValue(lowRankTensorShape, desiredRank, false);
    }

    public static long[] calculateShapeForLengthOneBroadcast(long[] shape1, long[] shape2) {
        return (shape1.length >= shape2.length) ? shape1 : shape2;
    }

    public static long[] getBroadcastResultShape(long[] left, long[] right) {

        final long[] shapeOfHighestRank = left.length > right.length ? left : right;
        long[] resultShape = Arrays.copyOf(shapeOfHighestRank, shapeOfHighestRank.length);

        int lowestRank = Math.min(left.length, right.length);
        for (int i = 1; i <= lowestRank; i++) {
            final long lDim = left[left.length - i];
            final long rDim = right[right.length - i];

            if (lDim != rDim && lDim != 1 && rDim != 1) {
                throw new IllegalArgumentException(
                    "Shape " + Arrays.toString(left) + " is not broadcastable with shape " + Arrays.toString(right)
                );
            }

            resultShape[resultShape.length - i] = Math.max(lDim, rDim);
        }

        return resultShape;
    }

    private static long[] increaseRankByPaddingValue(long[] lowRankTensorShape, int desiredRank, boolean append) {

        if (lowRankTensorShape.length == desiredRank) {
            return lowRankTensorShape;
        }

        if (lowRankTensorShape.length > desiredRank) {
            throw new IllegalArgumentException("low rank tensor must be rank less than or equal to desired rank");
        }

        long[] paddedShape = new long[desiredRank];

        Arrays.fill(paddedShape, 1);
        if (append) {
            System.arraycopy(lowRankTensorShape, 0, paddedShape, 0, lowRankTensorShape.length);
        } else {
            System.arraycopy(lowRankTensorShape, 0, paddedShape, paddedShape.length - lowRankTensorShape.length, lowRankTensorShape.length);
        }

        return paddedShape;
    }

    /**
     * It's possible to express negative dimensions, which are relative to the rank of a
     * tensor. E.g. given a rank 3 tensor, dimensions [-1, -2] would refer to the 3rd and 2nd dimension.
     *
     * @param rank       the rank that the dimension array is related to
     * @param dimensions positive dimensions are absolute and negative are relative to the rank
     * @return the dimensions converted to all absolute (positive). This mutates the passed in dimension argument.
     */
    public static int[] getAbsoluteDimensions(int rank, int[] dimensions) {
        for (int i = 0; i < dimensions.length; i++) {
            dimensions[i] = getAbsoluteDimension(dimensions[i], rank);
        }
        return dimensions;
    }

    /**
     * Removes a dimension from a shape. This will lower the rank by one.
     *
     * @param dimension the dimension to remove
     * @param shape     the shape to remove the dimension from
     * @return the shape without the given dimension
     * @throws IllegalArgumentException if the dimension does not exist
     */
    public static long[] removeDimension(int dimension, long[] shape) {
        TensorShapeValidation.checkDimensionExistsInShape(dimension, shape);
        return ArrayUtils.remove(shape, dimension);
    }

    /**
     * Finds the absolute dimension for a shape
     *
     * @param dimension the negative or positive dimension to find the absolute of
     * @param rank      the rank
     * @return an absolute dimension from a shape
     */
    public static int getAbsoluteDimension(int dimension, int rank) {
        if (dimension >= rank || dimension < -rank) {
            throw new IllegalArgumentException("Dimension " + dimension + " is invalid for rank " + rank + " tensor.");
        }
        if (dimension < 0) {
            dimension += rank;
        }
        return dimension;
    }

    public static long[] getSummationResultShape(long[] inputShape, int[] sumOverDimensions) {
        if (inputShape.length > 0) {
            return ArrayUtils.removeAll(inputShape, sumOverDimensions);
        } else {
            Preconditions.checkArgument(sumOverDimensions.length == 0);
            return inputShape;
        }
    }

    public static long[] getPermutedIndices(long[] indices, int... rearrange) {
        long[] permutedIndices = new long[indices.length];
        for (int i = 0; i < indices.length; i++) {
            permutedIndices[i] = indices[rearrange[i]];
        }
        return permutedIndices;
    }

    public static int[] invertedPermute(int[] rearrange) {
        int[] inverted = new int[rearrange.length];

        for (int i = 0; i < rearrange.length; i++) {
            inverted[rearrange[i]] = i;
        }

        return inverted;
    }

    public static int convertFromFlatIndexToPermutedFlatIndex(int fromFlatIndex,
                                                              long[] shape, long[] stride,
                                                              long[] permutedShape, long[] permutedStride,
                                                              int[] rearrange) {
        long[] shapeIndices = getShapeIndices(shape, stride, fromFlatIndex);

        long[] permutedIndex = getPermutedIndices(shapeIndices, rearrange);

        return Ints.checkedCast(getFlatIndex(permutedShape, permutedStride, permutedIndex));
    }

    /**
     * @param oldShape       The original shape to reshape from. This should have a shape length that
     *                       matches oldShapeLength
     * @param oldShapeLength The length of the old shape. e.g shape = [2, 2] then the length is 4
     * @param newShape       A shape that must be the same length as oldShape unless a single -1 dimension length
     *                       is specified. If -1 is used then a dimension length will be calculated in order to ensure
     *                       the new shape length is equal to the old shape length.
     * @return a copy of newShape if there is no wildcard used or a wildcard free shape with a length that matches
     * the oldShapeLength
     */
    public static long[] getReshapeAllowingWildcard(long[] oldShape, int oldShapeLength, long[] newShape) {
        long newLength = 1;
        int negativeDimension = -1;
        long[] newShapeCopy = new long[newShape.length];
        System.arraycopy(newShape, 0, newShapeCopy, 0, newShape.length);

        for (int i = 0; i < newShapeCopy.length; i++) {

            long dimILength = newShapeCopy[i];
            if (dimILength > 0) {
                newLength *= dimILength;
            } else if (dimILength < 0) {
                if (negativeDimension >= 0) {
                    throw new IllegalArgumentException("Cannot reshape " + Arrays.toString(oldShape) + " to " + Arrays.toString(newShapeCopy));
                }
                negativeDimension = i;
            }
        }

        if (newLength != oldShapeLength || negativeDimension >= 0) {
            if (negativeDimension < 0) {
                throw new IllegalArgumentException("Cannot reshape " + Arrays.toString(oldShape) + " to " + Arrays.toString(newShapeCopy));
            } else {
                newShapeCopy[negativeDimension] = oldShapeLength / newLength;
            }
        }

        return newShapeCopy;
    }

    public static long[] getConcatResultShape(int dimension, Tensor... toConcat) {
        Preconditions.checkArgument(toConcat.length > 0);

        Tensor first = toConcat[0];
        long[] firstShape = first.getShape();

        if (firstShape.length == 0 && dimension != 0) {
            throw new IllegalArgumentException("Cannot concat scalars on dimension " + dimension);
        }

        long[] concatShape = firstShape.length == 0 ? new long[]{1} : Arrays.copyOf(firstShape, firstShape.length);

        for (int i = 1; i < toConcat.length; i++) {
            Tensor c = toConcat[i];

            long[] cShape = c.getShape();
            for (int dim = 0; dim < concatShape.length; dim++) {

                if (dim == dimension) {
                    concatShape[dimension] += cShape.length == 0 ? 1 : cShape[dimension];
                } else {
                    if (cShape[dim] != concatShape[dim]) {
                        throw new IllegalArgumentException("Cannot concat shape " + Arrays.toString(cShape));
                    }
                }
            }
        }

        return concatShape;
    }

    public static int[] getPermutationForDimensionToDimensionZero(int dimension, long[] shape) {

        int[] rearrange = new int[shape.length];
        rearrange[0] = dimension;
        for (int i = 1; i < rearrange.length; i++) {
            if (i > dimension) {
                rearrange[i] = i;
            } else {
                rearrange[i] = i - 1;
            }
        }
        return rearrange;
    }

    public static int getBroadcastedFlatIndex(int fromFlatIndex, long[] fromStride, long[] toShape, long[] toStride) {

        final long[] fromShapeIndex = new long[fromStride.length];
        final long[] toShapeIndex = new long[fromShapeIndex.length];
        int remainder = fromFlatIndex;
        int toFlatIndex = 0;

        for (int i = 0; i < fromStride.length; i++) {
            fromShapeIndex[i] = remainder / fromStride[i];
            remainder -= fromShapeIndex[i] * fromStride[i];
            toShapeIndex[i] = fromShapeIndex[i] % toShape[i];
            toFlatIndex += toStride[i] * toShapeIndex[i];
        }

        return toFlatIndex;
    }

    public static boolean incrementIndexByShape(long[] shape, long[] index, int[] dimensionOrder) {

        for (int i : dimensionOrder) {
            index[i] = (index[i] + 1) % shape[i];
            if (index[i] != 0) {
                return true;
            }
        }

        return false;
    }

}
