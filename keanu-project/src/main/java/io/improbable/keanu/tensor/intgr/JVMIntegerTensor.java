package io.improbable.keanu.tensor.intgr;

import com.google.common.primitives.Ints;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.Arrays;
import java.util.function.Function;

import static java.util.Arrays.copyOf;

public class JVMIntegerTensor implements IntegerTensor {

    private int[] buffer;
    private long[] shape;
    private long[] stride;

    private JVMIntegerTensor(int value) {
        this.shape = new long[0];
        this.stride = new long[0];
        this.buffer = new int[]{value};
    }

    private JVMIntegerTensor(int[] data, long[] shape) {
        if (data.length != TensorShape.getLength(shape)) {
            throw new IllegalArgumentException(
                "Shape " + Arrays.toString(shape) + " does not match buffer size " + data.length);
        }

        this.shape = shape;
        this.stride = TensorShape.getRowFirstStride(shape);
        this.buffer = data;
    }

    private JVMIntegerTensor(int[] data, long[] shape, long[] stride) {
        if (data.length != TensorShape.getLength(shape)) {
            throw new IllegalArgumentException(
                "Shape " + Arrays.toString(shape) + " does not match buffer size" + data.length);
        }

        if (shape.length != stride.length) {
            throw new IllegalArgumentException(
                "Shape & Stride length don't match: (" + shape.length + ", " + stride.length + ")");
        }

        this.shape = shape;
        this.stride = stride;
        this.buffer = data;
    }

    public static JVMIntegerTensor scalar(int scalarValue) {
        return new JVMIntegerTensor(scalarValue);
    }

    public static JVMIntegerTensor create(int[] values, long... shape) {
        return new JVMIntegerTensor(values, shape);
    }

    public static JVMIntegerTensor create(int value, long... shape) {
        long length = TensorShape.getLength(shape);
        int[] buffer = new int[Ints.checkedCast(length)];

        if (value != 0) {
            Arrays.fill(buffer, value);
        }

        return new JVMIntegerTensor(buffer, shape);
    }

    public static JVMIntegerTensor ones(long... shape) {
        return create(1, shape);
    }

    public static JVMIntegerTensor zeros(long... shape) {
        return create(0, shape);
    }

    public static JVMIntegerTensor eye(long n) {

        int[] buffer = new int[Ints.checkedCast(n * n)];
        int nInt = Ints.checkedCast(n);
        for (int i = 0; i < n; i++) {
            buffer[i * nInt + i] = 1;
        }
        return new JVMIntegerTensor(buffer, new long[]{n, n});
    }

    private int[] copyBuffer() {
        return copyOf(buffer, buffer.length);
    }

    private long[] copyShape() {
        return copyOf(shape, shape.length);
    }

    private long[] copyStride() {
        return copyOf(stride, stride.length);
    }

    @Override
    public IntegerTensor setValue(Integer value, long... index) {
        return null;
    }

    @Override
    public IntegerTensor reshape(long... newShape) {
        return null;
    }

    @Override
    public IntegerTensor duplicate() {
        return new JVMIntegerTensor(copyBuffer(), copyShape(), copyStride());
    }

    @Override
    public IntegerTensor diag() {
        return null;
    }

    @Override
    public IntegerTensor transpose() {
        return null;
    }

    @Override
    public IntegerTensor sum(int... overDimensions) {
        return null;
    }

    @Override
    public IntegerTensor minus(int value) {
        return null;
    }

    @Override
    public IntegerTensor plus(int value) {
        return null;
    }

    @Override
    public IntegerTensor times(int value) {
        return null;
    }

    @Override
    public IntegerTensor div(int value) {
        return null;
    }

    @Override
    public IntegerTensor pow(int exponent) {
        return null;
    }

    @Override
    public IntegerTensor minus(IntegerTensor that) {
        return null;
    }

    @Override
    public IntegerTensor plus(IntegerTensor that) {
        return null;
    }

    @Override
    public IntegerTensor times(IntegerTensor that) {
        return null;
    }

    @Override
    public IntegerTensor matrixMultiply(IntegerTensor value) {
        return null;
    }

    @Override
    public IntegerTensor tensorMultiply(IntegerTensor value, int[] dimLeft, int[] dimsRight) {
        return null;
    }

    @Override
    public IntegerTensor div(IntegerTensor that) {
        return null;
    }

    @Override
    public IntegerTensor unaryMinus() {
        return null;
    }

    @Override
    public IntegerTensor abs() {
        return null;
    }

    @Override
    public IntegerTensor getGreaterThanMask(IntegerTensor greaterThanThis) {
        return null;
    }

    @Override
    public IntegerTensor getGreaterThanOrEqualToMask(IntegerTensor greaterThanThis) {
        return null;
    }

    @Override
    public IntegerTensor getLessThanMask(IntegerTensor lessThanThis) {
        return null;
    }

    @Override
    public IntegerTensor getLessThanOrEqualToMask(IntegerTensor lessThanThis) {
        return null;
    }

    @Override
    public IntegerTensor setWithMaskInPlace(IntegerTensor mask, Integer value) {
        return null;
    }

    @Override
    public IntegerTensor setWithMask(IntegerTensor mask, Integer value) {
        return null;
    }

    @Override
    public IntegerTensor apply(Function<Integer, Integer> function) {
        return null;
    }

    @Override
    public IntegerTensor slice(int dimension, long index) {
        return null;
    }

    @Override
    public IntegerTensor minusInPlace(int value) {
        return null;
    }

    @Override
    public IntegerTensor plusInPlace(int value) {
        return null;
    }

    @Override
    public IntegerTensor timesInPlace(int value) {
        return null;
    }

    @Override
    public IntegerTensor divInPlace(int value) {
        return null;
    }

    @Override
    public IntegerTensor powInPlace(int exponent) {
        return null;
    }

    @Override
    public BooleanTensor lessThan(int value) {
        return null;
    }

    @Override
    public BooleanTensor lessThanOrEqual(int value) {
        return null;
    }

    @Override
    public BooleanTensor greaterThan(int value) {
        return null;
    }

    @Override
    public BooleanTensor greaterThanOrEqual(int value) {
        return null;
    }

    @Override
    public IntegerTensor minInPlace(IntegerTensor min) {
        return null;
    }

    @Override
    public IntegerTensor maxInPlace(IntegerTensor max) {
        return null;
    }

    @Override
    public int min() {
        return 0;
    }

    @Override
    public int max() {
        return 0;
    }

    @Override
    public Integer sum() {
        return null;
    }

    @Override
    public DoubleTensor toDouble() {
        return null;
    }

    @Override
    public IntegerTensor toInteger() {
        return null;
    }

    @Override
    public IntegerTensor permute(int... rearrange) {
        return null;
    }

    @Override
    public int argMax() {
        return 0;
    }

    @Override
    public IntegerTensor argMax(int axis) {
        return null;
    }

    @Override
    public IntegerTensor minusInPlace(IntegerTensor that) {
        return null;
    }

    @Override
    public IntegerTensor plusInPlace(IntegerTensor that) {
        return null;
    }

    @Override
    public IntegerTensor timesInPlace(IntegerTensor that) {
        return null;
    }

    @Override
    public IntegerTensor divInPlace(IntegerTensor that) {
        return null;
    }

    @Override
    public IntegerTensor powInPlace(IntegerTensor exponent) {
        return null;
    }

    @Override
    public IntegerTensor unaryMinusInPlace() {
        return null;
    }

    @Override
    public IntegerTensor absInPlace() {
        return null;
    }

    @Override
    public IntegerTensor applyInPlace(Function<Integer, Integer> function) {
        return null;
    }

    @Override
    public BooleanTensor lessThan(IntegerTensor value) {
        return null;
    }

    @Override
    public BooleanTensor lessThanOrEqual(IntegerTensor value) {
        return null;
    }

    @Override
    public BooleanTensor greaterThan(IntegerTensor value) {
        return null;
    }

    @Override
    public BooleanTensor greaterThanOrEqual(IntegerTensor value) {
        return null;
    }

    @Override
    public IntegerTensor pow(IntegerTensor exponent) {
        return null;
    }

    @Override
    public int getRank() {
        return 0;
    }

    @Override
    public long[] getShape() {
        return new long[0];
    }

    @Override
    public long[] getStride() {
        return new long[0];
    }

    @Override
    public long getLength() {
        return 0;
    }

    @Override
    public Integer getValue(long... index) {
        return null;
    }

    @Override
    public Integer scalar() {
        return null;
    }

    @Override
    public double[] asFlatDoubleArray() {
        return new double[0];
    }

    @Override
    public int[] asFlatIntegerArray() {
        return new int[0];
    }

    @Override
    public Integer[] asFlatArray() {
        return new Integer[0];
    }

    @Override
    public FlattenedView<Integer> getFlattenedView() {
        return null;
    }

    @Override
    public BooleanTensor elementwiseEquals(Integer value) {
        return null;
    }
}
