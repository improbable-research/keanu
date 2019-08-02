package io.improbable.keanu.tensor;


import com.google.common.base.Preconditions;
import com.google.common.primitives.Longs;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.generic.GenericTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.tensor.jvm.Slicer;
import org.apache.commons.lang3.ArrayUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static io.improbable.keanu.tensor.TensorShape.getAbsoluteDimension;

public interface Tensor<N, T extends Tensor<N, T>> {

    static <DATA, TENSOR extends Tensor<DATA, TENSOR>> TENSOR scalar(DATA data) {
        if (data instanceof Double) {
            return (TENSOR) DoubleTensor.scalar(((Double) data).doubleValue());
        } else if (data instanceof Integer) {
            return (TENSOR) IntegerTensor.scalar(((Integer) data).intValue());
        } else if (data instanceof Boolean) {
            return (TENSOR) BooleanTensor.scalar(((Boolean) data).booleanValue());
        } else {
            return (TENSOR) GenericTensor.scalar(data);
        }
    }

    static <DATA, TENSOR extends Tensor<DATA, TENSOR>> TENSOR createFilled(DATA data, long[] shape) {
        if (data instanceof Double) {
            return (TENSOR) DoubleTensor.create(((Double) data).doubleValue(), shape);
        } else if (data instanceof Integer) {
            return (TENSOR) IntegerTensor.create(((Integer) data).intValue(), shape);
        } else if (data instanceof Boolean) {
            return (TENSOR) BooleanTensor.create(((Boolean) data).booleanValue(), shape);
        } else {
            return (TENSOR) GenericTensor.createFilled(data, shape);
        }
    }

    static <DATA, TENSOR extends Tensor<DATA, TENSOR>> TENSOR create(DATA[] data, long[] shape) {
        if (data instanceof Double[]) {
            return (TENSOR) DoubleTensor.create(ArrayUtils.toPrimitive((Double[]) data), shape);
        } else if (data instanceof Integer[]) {
            return (TENSOR) IntegerTensor.create(ArrayUtils.toPrimitive(((Integer[]) data)), shape);
        } else if (data instanceof Boolean[]) {
            return (TENSOR) BooleanTensor.create(ArrayUtils.toPrimitive(((Boolean[]) data)), shape);
        } else {
            return (TENSOR) GenericTensor.create(data, shape);
        }
    }

    static BooleanTensor elementwiseEquals(Tensor a, Tensor b) {
        if (!a.hasSameShapeAs(b)) {
            throw new IllegalArgumentException(
                String.format("Cannot compare tensors of different shapes %s and %s",
                    Arrays.toString(a.getShape()), Arrays.toString(b.getShape()))
            );
        }

        Object[] aArray = a.asFlatArray();
        Object[] bArray = b.asFlatArray();

        boolean[] equality = new boolean[aArray.length];

        for (int i = 0; i < aArray.length; i++) {
            equality[i] = aArray[i].equals(bArray[i]);
        }

        long[] shape = a.getShape();
        return BooleanTensor.create(equality, Arrays.copyOf(shape, shape.length));
    }

    long[] SCALAR_SHAPE = new long[]{};
    long[] SCALAR_STRIDE = new long[]{};
    long[] ONE_BY_ONE_SHAPE = new long[]{1, 1};

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
     * getValue returns a single primitive value from a specified index. The number of indices supplied must
     * match the rank of the tensor.
     *
     * @param index the index of the scalar value.
     * @return The primitive value at the specified index
     */
    default N getValue(long... index) {
        if (index.length == 1) {
            return getFlattenedView().get(index[0]);
        } else {
            return getFlattenedView().get(TensorShape.getFlatIndex(getShape(), getStride(), index));
        }
    }

    /**
     * @param booleanIndex a boolean tensor the same shape as this tensor where true is specified if the element
     *                     should be kept and false if not.
     * @return a vector with the values that align with true from the boolean index.
     */
    T get(BooleanTensor booleanIndex);

    default void setValue(N value, long... index) {
        if (index.length == 1) {
            getFlattenedView().set(index[0], value);
        } else {
            getFlattenedView().set(TensorShape.getFlatIndex(getShape(), getStride(), index), value);
        }
    }

    default N scalar() {
        if (this.getLength() > 1) {
            throw new IllegalArgumentException("Not a scalar");
        }
        return getValue(0);
    }

    T duplicate();

    T slice(int dimension, long index);

    default T slice(String sliceArg) {
        return slice(Slicer.fromString(sliceArg));
    }

    T slice(Slicer slicer);

    T take(long... index);

    /**
     * @param dimension      the dimension to split on
     * @param splitAtIndices the indices that the dimension to split on should be split on
     * @return pieces of the tensor split in the order specified by splitAtIndices. To get
     * pieces that encompasses the entire tensor, the last index in the splitAtIndices must
     * be the length of the dimension being split on.
     * <p>
     * e.g A =
     * [
     * 1, 2, 3, 4, 5, 6
     * 7, 8, 9, 1, 2, 3
     * ]
     * <p>
     * A.split(0, [1]) gives List([1, 2, 3, 4, 5, 6])
     * A.split(0, [1, 2]) gives List([1, 2, 3, 4, 5, 6], [7, 8, 9, 1, 2, 3]
     * <p>
     * A.split(1, [1, 3, 6]) gives
     * List(
     * [1, [2, 3  , [4, 5, 6,
     * 7]  8, 9]    1, 2, 3]
     * )
     */
    List<T> split(int dimension, long... splitAtIndices);

    default List<T> sliceAlongDimension(int dimension, long indexStart, long indexEnd) {
        List<T> slicedTensors = new ArrayList<>();

        for (long i = indexStart; i < indexEnd; i++) {
            slicedTensors.add(slice(dimension, i));
        }

        return slicedTensors;
    }

    T diag();

    default T transpose() {
        Preconditions.checkArgument(
            getRank() == 2,
            "Can only transpose rank 2. Use permute(...) for higher rank transpose."
        );
        return permute(1, 0);
    }

    N[] asFlatArray();

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

    T permute(int... rearrange);

    T broadcast(long... toShape);

    FlattenedView<N> getFlattenedView();

    interface FlattenedView<N> {

        long size();

        N get(long index);

        N getOrScalar(long index);

        void set(long index, N value);
    }

    default List<N> asFlatList() {
        return Arrays.asList(asFlatArray());
    }

    default boolean isLengthOne() {
        return getLength() == 1;
    }

    default boolean isScalar() {
        return getRank() == 0;
    }

    /**
     * Returns true if the tensor is a vector. A vector being a 1xn or a nx1 tensor.
     * <p>
     * (1, 2, 3) is a 1x3 vector.
     * <p>
     * (1)
     * (2)
     * (3) is a 3x1 vector.
     *
     * @return true if the tensor is a vector
     */
    default boolean isVector() {
        return getRank() == 1;
    }

    default boolean isMatrix() {
        return getRank() == 2;
    }

    default boolean hasSameShapeAs(Tensor that) {
        return hasSameShapeAs(that.getShape());
    }

    default boolean hasSameShapeAs(long[] shape) {
        return Arrays.equals(this.getShape(), shape);
    }

    default BooleanTensor elementwiseEquals(Tensor that) {
        return elementwiseEquals(this, that);
    }

    BooleanTensor elementwiseEquals(N value);

}
