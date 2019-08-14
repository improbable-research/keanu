package io.improbable.keanu;

import com.google.common.base.Preconditions;
import com.google.common.primitives.Longs;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.jvm.Slicer;
import org.apache.commons.lang3.ArrayUtils;

import java.util.ArrayList;
import java.util.List;

import static io.improbable.keanu.tensor.TensorShape.getAbsoluteDimension;

public interface BaseTensor<
    BOOLEAN extends BaseTensor<BOOLEAN, Boolean, BOOLEAN>,
    N,
    T extends BaseTensor<BOOLEAN, N, T>
    > {

    long[] SCALAR_SHAPE = new long[]{};

    int getRank();

    long[] getShape();

    /**
     * Returns the stride for each dimension of the tensor (based on C ordering).
     * <p>
     * The stride is the distance you'd move in a flat representation of the tensor for each index within that dimension
     * EG) For a 2x2 Tensor the Tensor would be laid out (in C order):
     * [{0, 0}, {0, 1}, {1, 0}, {1, 1}]
     * Thus the stride array would be provided as:
     * [2, 1]
     *
     * @return The stride array for this tensor
     */
    long[] getStride();

    long getLength();

    /**
     * @param booleanIndex a boolean tensor the same shape as this tensor where true is specified if the element
     *                     should be kept and false if not.
     * @return a vector with the values that align with true from the boolean index.
     */
    T get(BOOLEAN booleanIndex);

    T slice(int dimension, long index);

    default T slice(String sliceArg) {
        return slice(Slicer.fromString(sliceArg));
    }

    T slice(Slicer slicer);

    T take(long... index);

    default List<T> sliceAlongDimension(int dimension, long indexStart, long indexEnd) {
        List<T> slicedTensors = new ArrayList<>();

        for (long i = indexStart; i < indexEnd; i++) {
            slicedTensors.add(slice(dimension, i));
        }

        return slicedTensors;
    }

    T diag();

    T diagPart();

    /**
     * Fills a square matrix upper or lower triangle given a vector of values. The vector length must be sufficient to
     * fill a triangle of a square matrix. For a square matrix with rows and columns counts of N, the vector length
     * will need to be N*(N-1)/2.
     * <p>
     * E.g. Given [1,2,3,4,5,6]
     * <p>
     * Fill upper triangle will return
     * <p>
     * [1,2,3]
     * [0,4,5]
     * [0,0,6]
     * <p>
     * Fill lower triangle will return the transpose of fill upper.
     * <p>
     * [1,0,0]
     * [2,4,0]
     * [3,5,6]
     *
     * @param fillUpper fill the upper triangle if true
     * @param fillLower fill the lower triangle if false
     * @return a matrix with the upper and/or lower triangular filled with these values
     */
    T fillTriangular(boolean fillUpper, boolean fillLower);

    /**
     * The upper triangle or lower triangle of a square matrix vectorized. The upper triangle is vectorized by row
     * and the lower triangle is vectorized by column. This can be used to reverse a fillTriangular.
     * E.g Given
     * <p>
     * [1,2,3]
     * [4,5,6]
     * [7,8,9]
     * <p>
     * Upper part would return
     * <p>
     * [1,2,3,5,6,9]
     * <p>
     * Lower part would return
     * <p>
     * [1,4,7,5,8,9]
     *
     * @param upperPart true if the upper triangle should be taken of false for the lower triangle
     * @return the upper or lower triangle vectorized
     */
    T trianglePart(boolean upperPart);

    /**
     * Copies the upper triangle of a matrix to a new matrix of the same shape. This does not need to be a
     * square matrix.
     * <p>
     * E.g. Given
     * <p>
     * [1,2,3]
     * [4,5,6]
     * [7,8,9]
     * <p>
     * tri upper with a k=0 returns
     * <p>
     * [1,2,3]
     * [0,5,6]
     * [0,0,9]
     *
     * @param k An offset that controls how far from the matrix diagonal to copy the upper triangle. 0 is take the
     *          upper triangle starting at the diagonal. 1 is the upper triangle excluding the matrix diagonal. -1
     *          is the upper triangle and the first values of the lower triangle nearest to the diagonal.
     * @return a new matrix of the same shape with the lower triangle values missing.
     */
    T triUpper(int k);


    /**
     * Copies the lower triangle of a matrix to a new matrix of the same shape. This does not need to be a
     * square matrix.
     * <p>
     * E.g. Given
     * <p>
     * [1,2,3]
     * [4,5,6]
     * [7,8,9]
     * <p>
     * tri lower with a k=0 returns
     * <p>
     * [1,0,0]
     * [4,5,0]
     * [7,8,9]
     *
     * @param k An offset that controls how far from the matrix diagonal to copy the lower triangle. 0 is take the
     *          lower triangle starting at the diagonal. 1 is the lower triangle excluding the matrix diagonal. -1
     *          is the lower triangle and the first values of the upper triangle nearest to the diagonal.
     * @return a new matrix of the same shape with the upper triangle values missing.
     */
    T triLower(int k);

    default T transpose() {
        Preconditions.checkArgument(
            getRank() == 2,
            "Can only transpose rank 2. Use permute(...) for higher rank transpose."
        );
        return permute(1, 0);
    }

    T reshape(long... newShape);

    default T squeeze() {
        final long[] shape = getShape();
        List<Long> squeezedShape = new ArrayList<>();
        for (long length : shape) {
            if (length > 1) {
                squeezedShape.add(length);
            }
        }
        return reshape(Longs.toArray(squeezedShape));
    }

    default T expandDims(int axis) {
        final long[] shape = getShape();
        return reshape(ArrayUtils.insert(axis, shape, 1L));
    }

    default T moveAxis(int source, int destination) {

        int[] dimensionRange = TensorShape.dimensionRange(0, getRank());
        source = getAbsoluteDimension(source, dimensionRange.length);
        destination = getAbsoluteDimension(destination, dimensionRange.length);

        int[] rearrange = ArrayUtils.insert(destination, ArrayUtils.remove(dimensionRange, source), source);

        return permute(rearrange);
    }

    default T swapAxis(int axis1, int axis2) {

        int[] rearrange = TensorShape.dimensionRange(0, getRank());
        axis1 = getAbsoluteDimension(axis1, rearrange.length);
        axis2 = getAbsoluteDimension(axis2, rearrange.length);

        final int temp = rearrange[axis1];
        rearrange[axis1] = axis2;
        rearrange[axis2] = temp;

        return permute(rearrange);
    }

    T where(BOOLEAN predicate, T els);

    T permute(int... rearrange);

    T broadcast(long... toShape);

    default boolean isLengthOne() {
        return getLength() == 1;
    }

    /**
     * @return true if the tensor is rank 0
     */
    default boolean isScalar() {
        return getRank() == 0;
    }

    /**
     * @return true if the tensor is rank 1
     */
    default boolean isVector() {
        return getRank() == 1;
    }

    /**
     * @return true if the tensor is rank 2
     */
    default boolean isMatrix() {
        return getRank() == 2;
    }

    BOOLEAN elementwiseEquals(T that);

    BOOLEAN elementwiseEquals(N value);

    BOOLEAN notEqualTo(T that);

    BOOLEAN notEqualTo(N value);
}
