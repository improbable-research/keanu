package io.improbable.keanu.tensor.bool;

import com.google.common.base.Preconditions;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import org.apache.commons.lang3.ArrayUtils;

import java.util.Arrays;

import static com.google.common.primitives.Ints.checkedCast;
import static io.improbable.keanu.tensor.TensorShape.getFlatIndex;
import static java.util.Arrays.copyOf;

public class SimpleBooleanTensor implements BooleanTensor {

    static BooleanTensor create(boolean[] data, long[] shape) {
        return new SimpleBooleanTensor(data, shape);
    }

    static BooleanTensor scalar(boolean value) {
        return new SimpleBooleanTensor(value);
    }

    private final boolean[] data;
    private final long[] shape;
    private final long[] stride;

    /**
     * @param data  tensor data used c ordering
     * @param shape desired shape of tensor
     */
    public SimpleBooleanTensor(boolean[] data, long[] shape) {
        Preconditions.checkArgument(
            TensorShape.getLength(shape) == data.length,
            "Shape " + Arrays.toString(shape) + " does not match data length " + data.length
        );
        this.data = new boolean[data.length];
        System.arraycopy(data, 0, this.data, 0, this.data.length);
        this.shape = Arrays.copyOf(shape, shape.length);
        this.stride = TensorShape.getRowFirstStride(shape);
    }

    /**
     * @param constant constant boolean value to fill shape
     */
    public SimpleBooleanTensor(boolean constant) {
        this.data = new boolean[]{constant};
        this.shape = Tensor.SCALAR_SHAPE;
        this.stride = Tensor.SCALAR_STRIDE;
    }

    /**
     * @param shape shape to use as place holder
     */
    public SimpleBooleanTensor(long[] shape) {
        this.data = null;
        this.shape = Arrays.copyOf(shape, shape.length);
        this.stride = TensorShape.getRowFirstStride(shape);
    }

    /**
     * @param constant constant boolean value to fill shape
     * @param shape    desired shape of tensor
     */
    public SimpleBooleanTensor(boolean constant, long[] shape) {
        int dataLength = TensorShape.getLengthAsInt(shape);
        this.data = new boolean[dataLength];
        Arrays.fill(this.data, constant);
        this.shape = Arrays.copyOf(shape, shape.length);
        this.stride = TensorShape.getRowFirstStride(shape);
    }

    @Override
    public BooleanTensor reshape(long... newShape) {
        if (TensorShape.getLength(shape) != TensorShape.getLength(newShape)) {
            throw new IllegalArgumentException("Cannot reshape a tensor to a shape of different length. Failed to reshape: "
                + Arrays.toString(shape) + " to: " + Arrays.toString(newShape));
        }
        return new SimpleBooleanTensor(data, newShape);
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
    public BooleanTensor xor(BooleanTensor that) {
        return duplicate().xorInPlace(that);
    }

    @Override
    public BooleanTensor not() {
        return duplicate().notInPlace();
    }

    @Override
    public DoubleTensor doubleWhere(DoubleTensor trueValue, DoubleTensor falseValue) {
        double[] trueValues = trueValue.asFlatDoubleArray();
        double[] falseValues = falseValue.asFlatDoubleArray();

        double[] result = new double[data.length];
        for (int i = 0; i < result.length; i++) {
            result[i] = data[i] ? getOrScalar(trueValues, i) : getOrScalar(falseValues, i);
        }

        return DoubleTensor.create(result, copyOf(shape, shape.length));
    }

    private double getOrScalar(double[] values, int index) {
        if (values.length == 1) {
            return values[0];
        } else {
            return values[index];
        }
    }

    @Override
    public IntegerTensor integerWhere(IntegerTensor trueValue, IntegerTensor falseValue) {
        FlattenedView<Integer> trueValuesFlattened = trueValue.getFlattenedView();
        FlattenedView<Integer> falseValuesFlattened = falseValue.getFlattenedView();

        int[] result = new int[data.length];
        for (int i = 0; i < result.length; i++) {
            result[i] = data[i] ? trueValuesFlattened.getOrScalar(i) : falseValuesFlattened.getOrScalar(i);
        }

        return IntegerTensor.create(result, copyOf(shape, shape.length));
    }

    @Override
    public BooleanTensor booleanWhere(BooleanTensor trueValue, BooleanTensor falseValue) {
        FlattenedView<Boolean> trueValuesFlattened = trueValue.getFlattenedView();
        FlattenedView<Boolean> falseValuesFlattened = falseValue.getFlattenedView();

        boolean[] result = new boolean[data.length];
        for (int i = 0; i < result.length; i++) {
            result[i] = data[i] ? trueValuesFlattened.getOrScalar(i) : falseValuesFlattened.getOrScalar(i);
        }

        return BooleanTensor.create(result, copyOf(shape, shape.length));
    }

    @Override
    public <T, TENSOR extends Tensor<T>> TENSOR where(TENSOR trueValue, TENSOR falseValue) {
        if (trueValue instanceof DoubleTensor && falseValue instanceof DoubleTensor) {
            return (TENSOR) doubleWhere((DoubleTensor) trueValue, (DoubleTensor) falseValue);
        } else if (trueValue instanceof IntegerTensor && falseValue instanceof IntegerTensor) {
            return (TENSOR) integerWhere((IntegerTensor) trueValue, (IntegerTensor) falseValue);
        } else if (trueValue instanceof BooleanTensor && falseValue instanceof BooleanTensor) {
            return (TENSOR) booleanWhere((BooleanTensor) trueValue, (BooleanTensor) falseValue);
        } else {
            FlattenedView<T> trueValuesFlattened = trueValue.getFlattenedView();
            FlattenedView<T> falseValuesFlattened = falseValue.getFlattenedView();

            T[] result = (T[]) (new Object[data.length]);
            for (int i = 0; i < result.length; i++) {
                result[i] = data[i] ? trueValuesFlattened.getOrScalar(i) : falseValuesFlattened.getOrScalar(i);
            }

            return Tensor.create(result, copyOf(shape, shape.length));
        }
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
    public BooleanTensor xorInPlace(BooleanTensor that) {
        Boolean[] thatData = that.asFlatArray();
        for (int i = 0; i < data.length; i++) {
            data[i] = data[i] ^ thatData[i];
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
    public BooleanTensor slice(int dimension, long index) {
        DoubleTensor tadDoubles = Nd4jDoubleTensor.create(asFlatDoubleArray(), shape).slice(dimension, index);
        double[] tadFlat = tadDoubles.asFlatDoubleArray();
        boolean[] tadToBooleans = new boolean[tadFlat.length];
        for (int i = 0; i < tadFlat.length; i++) {
            tadToBooleans[i] = tadFlat[i] == 1;
        }
        return new SimpleBooleanTensor(tadToBooleans, tadDoubles.getShape());
    }

    @Override
    public int getRank() {
        return shape.length;
    }

    @Override
    public long[] getShape() {
        return Arrays.copyOf(shape, shape.length);
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
    public Boolean getValue(long... index) {
        return data[checkedCast(getFlatIndex(shape, stride, index))];
    }

    @Override
    public BooleanTensor setValue(Boolean value, long... index) {
        data[checkedCast(getFlatIndex(shape, stride, index))] = value;
        return this;
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
    public String toString() {

        StringBuilder dataString = new StringBuilder();
        if (data.length > 20) {
            dataString.append(Arrays.toString(Arrays.copyOfRange(data, 0, 10)));
            dataString.append("...");
            dataString.append(Arrays.toString(Arrays.copyOfRange(data, data.length - 10, data.length)));
        } else {
            dataString.append(Arrays.toString(data));
        }

        return "{\n" +
            "shape = " + Arrays.toString(shape) +
            "\ndata = " + dataString.toString() +
            "\n}";
    }

    @Override
    public FlattenedView<Boolean> getFlattenedView() {
        return new SimpleBooleanFlattenedView(data);
    }

    @Override
    public BooleanTensor elementwiseEquals(Boolean value) {
        return Tensor.elementwiseEquals(this, BooleanTensor.create(value, this.getShape()));
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

    @Override
    public boolean[] asFlatBooleanArray() {
        return data;
    }

}
