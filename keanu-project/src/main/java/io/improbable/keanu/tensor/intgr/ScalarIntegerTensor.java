package io.improbable.keanu.tensor.intgr;

import com.google.common.base.Preconditions;
import com.google.common.math.IntMath;
import io.improbable.keanu.tensor.Slicer;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import org.apache.commons.lang3.ArrayUtils;

import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

import static io.improbable.keanu.tensor.TensorShape.calculateShapeForLengthOneBroadcast;


public class ScalarIntegerTensor implements IntegerTensor {

    private Integer value;
    private long[] shape;

    private ScalarIntegerTensor(Integer value, long[] shape) {
        this.value = value;
        this.shape = shape;
    }

    public ScalarIntegerTensor(int value) {
        this(value, SCALAR_SHAPE);
    }

    public ScalarIntegerTensor(long[] shape) {
        this(null, shape);
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
    public long[] getStride() {
        return new long[shape.length];
    }

    @Override
    public long getLength() {
        return 1;
    }

    @Override
    public IntegerTensor duplicate() {
        return new ScalarIntegerTensor(value, shape);
    }

    @Override
    public Integer sum() {
        return value;
    }

    @Override
    public DoubleTensor toDouble() {
        return DoubleTensor.create(value, shape);
    }

    @Override
    public IntegerTensor toInteger() {
        return this;
    }

    @Override
    public Integer scalar() {
        return value;
    }

    @Override
    public IntegerTensor reshape(long... newShape) {
        if (TensorShape.getLength(newShape) != 1) {
            throw new IllegalArgumentException("Cannot reshape scalar to non scalar");
        }

        return new ScalarIntegerTensor(value, newShape);
    }

    @Override
    public IntegerTensor permute(int... rearrange) {
        if (rearrange.length > shape.length) {
            throw new IllegalArgumentException("Cannot permute " + Arrays.toString(rearrange) + " on shape " + Arrays.toString(shape));
        }
        return new ScalarIntegerTensor(value, shape);
    }

    @Override
    public IntegerTensor broadcast(long... toShape) {
        int bufferLength = TensorShape.getLengthAsInt(toShape);
        int[] buffer = new int[bufferLength];
        Arrays.fill(buffer, value);
        return IntegerTensor.create(buffer, toShape);
    }

    @Override
    public IntegerTensor diag() {
        return duplicate();
    }

    @Override
    public IntegerTensor transpose() {
        return duplicate();
    }

    @Override
    public IntegerTensor sum(int... overDimensions) {
        overDimensions = TensorShape.setToAbsoluteDimensions(shape.length, overDimensions);
        long[] summedShape = ArrayUtils.removeAll(shape, overDimensions);
        return new ScalarIntegerTensor(value, summedShape);
    }


    @Override
    public IntegerTensor cumSumInPlace(int dimension) {
        return this;
    }

    @Override
    public Integer product() {
        return value;
    }

    @Override
    public IntegerTensor product(int... overDimensions) {
        return this;
    }

    @Override
    public IntegerTensor cumProdInPlace(int dimension) {
        return this;
    }

    @Override
    public boolean equalsWithinEpsilon(IntegerTensor other, Integer epsilon) {
        if (other instanceof ScalarIntegerTensor) {
            return Math.abs(other.scalar() - value) < epsilon;
        } else {
            return other.equalsWithinEpsilon(this, epsilon);
        }
    }

    @Override
    public IntegerTensor matrixMultiply(IntegerTensor that) {
        if (that.isLengthOne()) {
            return that.times(value);
        }
        throw new IllegalArgumentException("Cannot use matrix multiply with scalar. Use times instead.");
    }

    @Override
    public IntegerTensor tensorMultiply(IntegerTensor that, int[] dimsLeft, int[] dimsRight) {
        if (that.isLengthOne()) {
            if (dimsLeft.length > 1 || dimsRight.length > 1 || dimsLeft[0] != 0 || dimsRight[0] != 0) {
                throw new IllegalArgumentException("Tensor multiply sum dimensions out of bounds for scalar");
            }
            return that.times(value);
        }
        throw new IllegalArgumentException("Cannot use tensor multiply with scalar. Use times instead.");
    }

    @Override
    public IntegerTensor greaterThanMask(IntegerTensor greaterThanThis) {
        if (greaterThanThis.isLengthOne()) {
            long[] newShape = calculateShapeForLengthOneBroadcast(shape, greaterThanThis.getShape());
            return new ScalarIntegerTensor(value > greaterThanThis.scalar() ? 1 : 0, newShape);
        } else {
            return IntegerTensor.create(value, greaterThanThis.getShape())
                .greaterThanMask(greaterThanThis);
        }
    }

    @Override
    public IntegerTensor greaterThanOrEqualToMask(IntegerTensor greaterThanOrEqualToThis) {
        if (greaterThanOrEqualToThis.isLengthOne()) {
            long[] newShape = calculateShapeForLengthOneBroadcast(shape, greaterThanOrEqualToThis.getShape());
            return new ScalarIntegerTensor(value >= greaterThanOrEqualToThis.scalar() ? 1 : 0, newShape);
        } else {
            return IntegerTensor.create(value, greaterThanOrEqualToThis.getShape())
                .greaterThanOrEqualToMask(greaterThanOrEqualToThis);
        }
    }

    @Override
    public IntegerTensor lessThanMask(IntegerTensor lessThanThis) {
        if (lessThanThis.isLengthOne()) {
            long[] newShape = calculateShapeForLengthOneBroadcast(shape, lessThanThis.getShape());
            return new ScalarIntegerTensor(value < lessThanThis.scalar() ? 1 : 0, newShape);
        } else {
            return IntegerTensor.create(value, lessThanThis.getShape())
                .lessThanMask(lessThanThis);
        }
    }

    @Override
    public IntegerTensor lessThanOrEqualToMask(IntegerTensor lessThanOrEqualsThis) {
        if (lessThanOrEqualsThis.isLengthOne()) {
            long[] newShape = calculateShapeForLengthOneBroadcast(shape, lessThanOrEqualsThis.getShape());
            return new ScalarIntegerTensor(value <= lessThanOrEqualsThis.scalar() ? 1 : 0, newShape);
        } else {
            return IntegerTensor.create(value, lessThanOrEqualsThis.getShape())
                .lessThanOrEqualToMask(lessThanOrEqualsThis);
        }
    }

    @Override
    public IntegerTensor setWithMaskInPlace(IntegerTensor withMask, Integer valueToApply) {
        if (withMask.isLengthOne()) {
            this.value = withMask.scalar() == 1.0 ? valueToApply : this.value;
        } else {
            return IntegerTensor.create(value, withMask.getShape())
                .setWithMaskInPlace(withMask, valueToApply);
        }
        return this;
    }

    @Override
    public IntegerTensor slice(int dimension, long index) {
        if (dimension == 0 && index == 0) {
            return duplicate();
        } else {
            throw new IllegalStateException("Slice is only valid for dimension and index zero in a scalar");
        }
    }

    @Override
    public IntegerTensor slice(Slicer slicer) {
        return slice(0, 0);
    }

    @Override
    public IntegerTensor take(long... index) {
        return new ScalarIntegerTensor(getValue(index));
    }

    @Override
    public List<IntegerTensor> split(int dimension, long... splitAtIndices) {
        throw new UnsupportedOperationException("Cannot split scalar!");
    }

    @Override
    public IntegerTensor minusInPlace(Integer that) {
        value = value - that;
        return this;
    }

    @Override
    public IntegerTensor plusInPlace(Integer that) {
        value = value + that;
        return this;
    }

    @Override
    public IntegerTensor timesInPlace(Integer that) {
        value = value * that;
        return this;
    }

    @Override
    public IntegerTensor divInPlace(Integer that) {
        value = value / that;
        return this;
    }

    @Override
    public IntegerTensor powInPlace(IntegerTensor exponent) {
        if (exponent.isLengthOne()) {
            powInPlace(exponent.scalar());
            shape = calculateShapeForLengthOneBroadcast(shape, exponent.getShape());
        } else {
            return IntegerTensor.create(value, exponent.getShape())
                .powInPlace(exponent);
        }
        return this;
    }

    @Override
    public IntegerTensor powInPlace(Integer exponent) {
        value = IntMath.pow(value, exponent);
        return this;
    }

    @Override
    public Integer average() {
        return value;
    }

    @Override
    public Integer standardDeviation() {
        return 0;
    }

    @Override
    public IntegerTensor minusInPlace(IntegerTensor that) {
        if (that.isLengthOne()) {
            minusInPlace(that.scalar());
            shape = calculateShapeForLengthOneBroadcast(shape, that.getShape());
        } else {
            return IntegerTensor.create(value, that.getShape())
                .minusInPlace(that);
        }
        return this;
    }

    @Override
    public IntegerTensor reverseMinusInPlace(IntegerTensor value) {
        if (value instanceof ScalarIntegerTensor) {
            return reverseMinusInPlace(value.scalar());
        } else {
            return value.minus(this.value);
        }
    }

    @Override
    public IntegerTensor reverseMinusInPlace(Integer value) {
        this.value = value - this.value;
        return this;
    }

    @Override
    public IntegerTensor plusInPlace(IntegerTensor that) {
        if (that.isLengthOne()) {
            plusInPlace(that.scalar());
            shape = calculateShapeForLengthOneBroadcast(shape, that.getShape());
        } else {
            return IntegerTensor.create(value, that.getShape())
                .plusInPlace(that);
        }
        return this;
    }

    @Override
    public IntegerTensor timesInPlace(IntegerTensor that) {
        if (that.isLengthOne()) {
            timesInPlace(that.scalar());
            shape = calculateShapeForLengthOneBroadcast(shape, that.getShape());
        } else {
            return IntegerTensor.create(value, that.getShape())
                .timesInPlace(that);
        }
        return this;
    }

    @Override
    public IntegerTensor divInPlace(IntegerTensor that) {
        if (that.isLengthOne()) {
            divInPlace(that.scalar());
            shape = calculateShapeForLengthOneBroadcast(shape, that.getShape());
        } else {
            return IntegerTensor.create(value, that.getShape())
                .divInPlace(that);
        }
        return this;
    }

    @Override
    public IntegerTensor reverseDivInPlace(Integer value) {
        this.value = value / this.value;
        return this;
    }

    @Override
    public IntegerTensor reverseDivInPlace(IntegerTensor value) {
        if (value instanceof ScalarIntegerTensor) {
            return reverseDivInPlace(value.scalar());
        } else {
            return value.div(this.value);
        }
    }

    @Override
    public IntegerTensor unaryMinusInPlace() {
        value = -value;
        return this;
    }

    @Override
    public IntegerTensor absInPlace() {
        value = Math.abs(value);
        return this;
    }

    @Override
    public IntegerTensor applyInPlace(Function<Integer, Integer> function) {
        value = function.apply(value);
        return this;
    }

    @Override
    public IntegerTensor setAllInPlace(Integer value) {
        this.value = value;
        return this;
    }

    @Override
    public IntegerTensor safeLogTimesInPlace(IntegerTensor y) {
        throw new UnsupportedOperationException();
    }

    @Override
    public BooleanTensor lessThan(Integer that) {
        return BooleanTensor.create(this.value < that, shape);
    }

    @Override
    public BooleanTensor lessThanOrEqual(Integer that) {
        return BooleanTensor.create(this.value <= that, shape);
    }

    @Override
    public BooleanTensor lessThan(IntegerTensor that) {
        if (that.isLengthOne()) {
            long[] newShape = calculateShapeForLengthOneBroadcast(shape, that.getShape());
            return BooleanTensor.create(this.value < that.scalar(), newShape);
        } else {
            return that.greaterThan(value);
        }
    }

    @Override
    public BooleanTensor lessThanOrEqual(IntegerTensor that) {
        if (that.isLengthOne()) {
            long[] newShape = calculateShapeForLengthOneBroadcast(shape, that.getShape());
            return BooleanTensor.create(this.value <= that.scalar(), newShape);
        } else {
            return that.greaterThanOrEqual(value);
        }
    }

    @Override
    public BooleanTensor greaterThan(Integer value) {
        return BooleanTensor.create(this.value > value, shape);
    }

    @Override
    public BooleanTensor greaterThanOrEqual(Integer value) {
        return BooleanTensor.create(this.value >= value, shape);
    }

    @Override
    public IntegerTensor minInPlace(IntegerTensor min) {
        if (min.isLengthOne()) {
            long[] newShape = calculateShapeForLengthOneBroadcast(shape, min.getShape());
            return new ScalarIntegerTensor(Math.min(value, min.scalar()), newShape);
        } else {
            return min.duplicate().minInPlace(this);
        }
    }

    @Override
    public IntegerTensor clampInPlace(IntegerTensor min, IntegerTensor max) {
        Preconditions.checkArgument(min.isScalar() && max.isScalar());
        return new ScalarIntegerTensor(Math.max(Math.min(value, max.scalar()), min.scalar()), shape);
    }

    @Override
    public IntegerTensor maxInPlace(IntegerTensor max) {
        if (max.isLengthOne()) {
            long[] newShape = calculateShapeForLengthOneBroadcast(shape, max.getShape());
            return new ScalarIntegerTensor(Math.max(value, max.scalar()), newShape);
        } else {
            return max.duplicate().maxInPlace(this);
        }
    }

    @Override
    public Integer min() {
        return value;
    }

    @Override
    public Integer max() {
        return value;
    }

    @Override
    public int argMax() {
        return 0;
    }

    @Override
    public IntegerTensor argMax(int axis) {
        TensorShapeValidation.checkDimensionExistsInShape(axis, this.getShape());
        return IntegerTensor.scalar(0);
    }

    @Override
    public int argMin() {
        return 0;
    }

    @Override
    public IntegerTensor argMin(int axis) {
        TensorShapeValidation.checkDimensionExistsInShape(axis, this.getShape());
        return IntegerTensor.scalar(0);
    }

    @Override
    public BooleanTensor greaterThan(IntegerTensor that) {
        if (that.isLengthOne()) {
            long[] newShape = calculateShapeForLengthOneBroadcast(shape, that.getShape());
            return BooleanTensor.create(this.value > that.scalar(), newShape);
        } else {
            return that.lessThan(value);
        }
    }

    @Override
    public BooleanTensor greaterThanOrEqual(IntegerTensor that) {
        if (that.isLengthOne()) {
            long[] newShape = calculateShapeForLengthOneBroadcast(shape, that.getShape());
            return BooleanTensor.create(this.value >= that.scalar(), newShape);
        } else {
            return that.lessThanOrEqual(value);
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Tensor)) return false;

        Tensor that = (Tensor) o;

        if (!Arrays.equals(that.getShape(), shape)) return false;
        return that.scalar().equals(value);
    }

    @Override
    public int hashCode() {
        int result = value != null ? value.hashCode() : 0;
        result = 31 * result + Arrays.hashCode(shape);
        return result;
    }

    @Override
    public FlattenedView<Integer> getFlattenedView() {
        return new SimpleIntegerFlattenedView();
    }

    @Override
    public BooleanTensor elementwiseEquals(Tensor that) {
        if (that instanceof IntegerTensor) {
            IntegerTensor thatAsInteger = (IntegerTensor) that;
            if (that.isLengthOne()) {
                long[] newShape = calculateShapeForLengthOneBroadcast(shape, that.getShape());
                return BooleanTensor.create(value.equals(thatAsInteger.scalar()), newShape);
            } else {
                return thatAsInteger.elementwiseEquals(value);
            }
        } else {
            return Tensor.elementwiseEquals(this, that);
        }
    }

    @Override
    public BooleanTensor elementwiseEquals(Integer value) {
        return BooleanTensor.create(this.scalar().equals(value), shape);
    }

    @Override
    public double[] asFlatDoubleArray() {
        return new double[]{value};
    }

    @Override
    public int[] asFlatIntegerArray() {
        return new int[]{value};
    }

    @Override
    public Integer[] asFlatArray() {
        return new Integer[]{value};
    }

    @Override
    public String toString() {
        return "ScalarIntegerTensor{" +
            "value=" + value +
            '}';
    }

    @Override
    public IntegerTensor modInPlace(Integer that) {
        value = value % that;
        return this;
    }

    @Override
    public IntegerTensor modInPlace(IntegerTensor that) {
        Preconditions.checkArgument(that.isScalar());
        return modInPlace(that.scalar());
    }

    private class SimpleIntegerFlattenedView implements FlattenedView<Integer> {

        @Override
        public long size() {
            return 1;
        }

        @Override
        public Integer get(long index) {
            if (index != 0) {
                throw new IndexOutOfBoundsException();
            }
            return value;
        }

        @Override
        public Integer getOrScalar(long index) {
            return value;
        }

        @Override
        public void set(long index, Integer value) {
            if (index != 0) {
                throw new IndexOutOfBoundsException();
            }
            ScalarIntegerTensor.this.value = value;
        }

    }

}
