package io.improbable.keanu.tensor;


import io.improbable.keanu.BaseTensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.generic.GenericTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import org.apache.commons.lang3.ArrayUtils;

import java.util.Arrays;
import java.util.List;

public interface Tensor<N, T extends Tensor<N, T>> extends BaseTensor<BooleanTensor, N, T> {

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

    N[] asFlatArray();

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

    BooleanTensor elementwiseEquals(T that);

    BooleanTensor elementwiseEquals(N value);

}
