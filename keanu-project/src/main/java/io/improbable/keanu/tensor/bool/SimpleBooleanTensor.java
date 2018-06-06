package io.improbable.keanu.tensor.bool;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.generic.GenericTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import org.apache.commons.lang3.ArrayUtils;

import java.util.Arrays;

import static io.improbable.keanu.tensor.TensorShape.getFlatIndex;
import static java.util.Arrays.copyOf;

public class SimpleBooleanTensor implements BooleanTensor {

    static BooleanTensor create(boolean[] data, int[] shape) {
        return new SimpleBooleanTensor(data, shape);
    }

    static BooleanTensor scalar(boolean value) {
        return new SimpleBooleanTensor(value);
    }

    private final boolean[] data;
    private final int[] shape;
    private final int[] stride;

    public SimpleBooleanTensor(boolean[] data, int[] shape) {
        this.data = data;
        this.shape = shape;
        this.stride = TensorShape.getRowFirstStride(shape);
    }

    public SimpleBooleanTensor(boolean constant) {
        this.data = new boolean[]{constant};
        this.shape = Tensor.SCALAR_SHAPE;
        this.stride = Tensor.SCALAR_STRIDE;
    }

    public SimpleBooleanTensor(int[] shape) {
        this.data = null;
        this.shape = shape;
        this.stride = TensorShape.getRowFirstStride(shape);
    }

    public SimpleBooleanTensor(boolean constant, int[] shape) {
        this.data = new boolean[(int) TensorShape.getLength(shape)];
        this.shape = shape;
        this.stride = TensorShape.getRowFirstStride(shape);
        Arrays.fill(this.data, constant);
    }

    @Override
    public BooleanTensor and(BooleanTensor that) {
        return duplicate().andInPlace(that);
    }

    @Override
    public BooleanTensor or(BooleanTensor that) {
        return duplicate().orInPlace(that);
    }

    @Override
    public BooleanTensor not() {
        return duplicate().notInPlace();
    }

    @Override
    public DoubleTensor setDoubleIf(DoubleTensor trueValue, DoubleTensor falseValue) {
        FlattenedView<Double> trueValuesFlattened = trueValue.getFlattenedView();
        FlattenedView<Double> falseValuesFlattened = falseValue.getFlattenedView();

        double[] result = new double[data.length];
        for (int i = 0; i < result.length; i++) {
            result[i] = data[i] ? trueValuesFlattened.getOrScalar(i) : falseValuesFlattened.getOrScalar(i);
        }

        return DoubleTensor.create(result, copyOf(shape, shape.length));
    }

    @Override
    public IntegerTensor setIntegerIf(IntegerTensor trueValue, IntegerTensor falseValue) {
        FlattenedView<Integer> trueValuesFlattened = trueValue.getFlattenedView();
        FlattenedView<Integer> falseValuesFlattened = falseValue.getFlattenedView();

        int[] result = new int[data.length];
        for (int i = 0; i < result.length; i++) {
            result[i] = data[i] ? trueValuesFlattened.getOrScalar(i) : falseValuesFlattened.getOrScalar(i);
        }

        return IntegerTensor.create(result, copyOf(shape, shape.length));
    }

    @Override
    public BooleanTensor setBooleanIf(BooleanTensor trueValue, BooleanTensor falseValue) {
        FlattenedView<Boolean> trueValuesFlattened = trueValue.getFlattenedView();
        FlattenedView<Boolean> falseValuesFlattened = falseValue.getFlattenedView();

        boolean[] result = new boolean[data.length];
        for (int i = 0; i < result.length; i++) {
            result[i] = data[i] ? trueValuesFlattened.getOrScalar(i) : falseValuesFlattened.getOrScalar(i);
        }

        return BooleanTensor.create(result, copyOf(shape, shape.length));
    }

    @Override
    public <T> Tensor<T> setIf(Tensor<T> trueValue, Tensor<T> falseValue) {
        FlattenedView<T> trueValuesFlattened = trueValue.getFlattenedView();
        FlattenedView<T> falseValuesFlattened = falseValue.getFlattenedView();

        T[] result = (T[]) (new Object[data.length]);
        for (int i = 0; i < result.length; i++) {
            result[i] = data[i] ? trueValuesFlattened.get(i) : falseValuesFlattened.get(i);
        }

        return new GenericTensor<>(result, copyOf(shape, shape.length));
    }

    @Override
    public BooleanTensor andInPlace(BooleanTensor that) {
        Boolean[] thatData = that.asFlatArray();
        for (int i = 0; i < data.length; i++) {
            data[i] = data[i] && thatData[i];
        }
        return this;
    }

    @Override
    public BooleanTensor orInPlace(BooleanTensor that) {
        Boolean[] thatData = that.asFlatArray();
        for (int i = 0; i < data.length; i++) {
            data[i] = data[i] || thatData[i];
        }
        return this;
    }

    @Override
    public BooleanTensor notInPlace() {
        for (int i = 0; i < data.length; i++) {
            data[i] = !data[i];
        }
        return this;
    }

    @Override
    public boolean allTrue() {
        for (int i = 0; i < data.length; i++) {
            if (!data[i]) {
                return false;
            }
        }
        return true;
    }

    @Override
    public boolean allFalse() {
        for (int i = 0; i < data.length; i++) {
            if (data[i]) {
                return false;
            }
        }
        return true;
    }

    @Override
    public DoubleTensor toDoubleMask() {
        double[] doubles = asFlatDoubleArray();
        return DoubleTensor.create(doubles, copyOf(shape, shape.length));
    }

    @Override
    public IntegerTensor toIntegerMask() {
        int[] doubles = asFlatIntegerArray();
        return IntegerTensor.create(doubles, copyOf(shape, shape.length));
    }

    @Override
    public int getRank() {
        return shape.length;
    }

    @Override
    public int[] getShape() {
        return shape;
    }

    @Override
    public long getLength() {
        return data.length;
    }

    @Override
    public boolean isShapePlaceholder() {
        return data == null;
    }

    @Override
    public Boolean getValue(int... index) {
        return data[getFlatIndex(shape, stride, index)];
    }

    @Override
    public void setValue(Boolean value, int... index) {
        data[getFlatIndex(shape, stride, index)] = value;
    }

    @Override
    public Boolean scalar() {
        return data[0];
    }

    @Override
    public BooleanTensor duplicate() {
        return new SimpleBooleanTensor(copyOf(data, data.length), copyOf(shape, shape.length));
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;

        if (o instanceof Tensor) {
            Tensor that = (Tensor) o;
            if (!Arrays.equals(that.getShape(), shape)) return false;
            return Arrays.equals(
                that.asFlatArray(),
                this.asFlatArray()
            );
        }

        return false;
    }

    @Override
    public int hashCode() {
        int result = Arrays.hashCode(data);
        result = 31 * result + Arrays.hashCode(shape);
        result = 31 * result + Arrays.hashCode(stride);
        return result;
    }

    @Override
    public FlattenedView<Boolean> getFlattenedView() {
        return new SimpleBooleanFlattenedView(data);
    }

    private static class SimpleBooleanFlattenedView implements FlattenedView<Boolean> {

        private boolean[] data;

        public SimpleBooleanFlattenedView(boolean[] data) {
            this.data = data;
        }

        @Override
        public long size() {
            return data.length;
        }

        @Override
        public Boolean get(long index) {
            if (index > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("Only integer based indexing supported for boolean tensors");
            }
            return data[(int) index];
        }

        @Override
        public Boolean getOrScalar(long index) {
            if (data.length == 1) {
                return get(0);
            } else {
                return get(index);
            }
        }

        @Override
        public void set(long index, Boolean value) {
            if (index > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("Only integer based indexing supported for generic tensors");
            }
            data[(int) index] = value;
        }

    }

    @Override
    public double[] asFlatDoubleArray() {
        double[] doubles = new double[data.length];
        for (int i = 0; i < doubles.length; i++) {
            doubles[i] = data[i] ? 1.0 : 0.0;
        }

        return doubles;
    }

    @Override
    public int[] asFlatIntegerArray() {
        int[] integers = new int[data.length];
        for (int i = 0; i < integers.length; i++) {
            integers[i] = data[i] ? 1 : 0;
        }

        return integers;
    }

    @Override
    public Boolean[] asFlatArray() {
        return ArrayUtils.toObject(data);
    }
}
