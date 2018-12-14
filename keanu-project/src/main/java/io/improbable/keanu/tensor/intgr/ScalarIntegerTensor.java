package io.improbable.keanu.tensor.intgr;

import com.google.common.math.IntMath;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import org.apache.commons.lang3.ArrayUtils;

import java.util.function.Function;


public class ScalarIntegerTensor implements IntegerTensor {

    private Integer value;

    public ScalarIntegerTensor(Integer value) {
        this.value = value;
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
    public long getLength() {
        return isShapePlaceholder() ? 0L : 1L;
    }

    @Override
    public boolean isShapePlaceholder() {
        return value == null;
    }

    @Override
    public IntegerTensor duplicate() {
        return new ScalarIntegerTensor(value);
    }

    @Override
    public Integer getValue(long... index) {
        if (index.length == 1 && index[0] == 0) {
            return value;
        } else {
            throw new IndexOutOfBoundsException(ArrayUtils.toString(index) + " out of bounds on scalar");
        }
    }

    @Override
    public IntegerTensor setValue(Integer value, long... index) {
        if (index.length == 1 && index[0] == 0) {
            this.value = value;
            return this;
        } else {
            throw new IndexOutOfBoundsException(ArrayUtils.toString(index) + " out of bounds on scalar");
        }
    }

    @Override
    public Integer sum() {
        return value;
    }

    @Override
    public DoubleTensor toDouble() {
        return DoubleTensor.create(value);
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

        if (newShape.length == 0) {
            return new ScalarIntegerTensor(value);
        } else {
            return Nd4jIntegerTensor.create(value, newShape);
        }
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
        return new ScalarIntegerTensor(value);
    }

    @Override
    public IntegerTensor minus(int that) {
        return duplicate().minusInPlace(that);
    }

    @Override
    public IntegerTensor plus(int that) {
        return duplicate().plusInPlace(that);
    }

    @Override
    public IntegerTensor times(int that) {
        return duplicate().timesInPlace(that);
    }

    @Override
    public IntegerTensor div(int that) {
        return duplicate().divInPlace(that);
    }

    @Override
    public IntegerTensor pow(IntegerTensor exponent) {
        return duplicate().powInPlace(exponent);
    }

    @Override
    public IntegerTensor pow(int exponent) {
        return duplicate().powInPlace(exponent);
    }

    @Override
    public IntegerTensor minus(IntegerTensor that) {
        return duplicate().minusInPlace(that);
    }

    @Override
    public IntegerTensor plus(IntegerTensor that) {
        return duplicate().plusInPlace(that);
    }

    @Override
    public IntegerTensor times(IntegerTensor that) {
        return duplicate().timesInPlace(that);
    }

    @Override
    public IntegerTensor matrixMultiply(IntegerTensor value) {
        if (value.isLengthOne()) {
            return value.times(value);
        }
        throw new IllegalArgumentException("Cannot use matrix multiply with scalar. Use times instead.");
    }

    @Override
    public IntegerTensor tensorMultiply(IntegerTensor value, int[] dimsLeft, int[] dimsRight) {
        if (value.isLengthOne()) {
            if (dimsLeft.length > 1 || dimsRight.length > 1 || dimsLeft[0] != 0 || dimsRight[0] != 0) {
                throw new IllegalArgumentException("Tensor multiply sum dimensions out of bounds for scalar");
            }
            return value.times(value);
        }
        throw new IllegalArgumentException("Cannot use tensor multiply with scalar. Use times instead.");
    }

    @Override
    public IntegerTensor div(IntegerTensor that) {
        return duplicate().divInPlace(that);
    }

    @Override
    public IntegerTensor unaryMinus() {
        return duplicate().unaryMinusInPlace();
    }

    @Override
    public IntegerTensor abs() {
        return duplicate().absInPlace();
    }

    @Override
    public IntegerTensor getGreaterThanMask(IntegerTensor greaterThanThis) {
        if (greaterThanThis.isLengthOne()) {
            return new ScalarIntegerTensor(value > greaterThanThis.scalar() ? 1 : 0);
        } else {
            return IntegerTensor.create(value, greaterThanThis.getShape())
                .getGreaterThanMask(greaterThanThis);
        }
    }

    @Override
    public IntegerTensor getGreaterThanOrEqualToMask(IntegerTensor greaterThanOrEqualToThis) {
        if (greaterThanOrEqualToThis.isLengthOne()) {
            return new ScalarIntegerTensor(value >= greaterThanOrEqualToThis.scalar() ? 1 : 0);
        } else {
            return IntegerTensor.create(value, greaterThanOrEqualToThis.getShape())
                .getGreaterThanOrEqualToMask(greaterThanOrEqualToThis);
        }
    }

    @Override
    public IntegerTensor getLessThanMask(IntegerTensor lessThanThis) {
        if (lessThanThis.isLengthOne()) {
            return new ScalarIntegerTensor(value < lessThanThis.scalar() ? 1 : 0);
        } else {
            return IntegerTensor.create(value, lessThanThis.getShape())
                .getLessThanMask(lessThanThis);
        }
    }

    @Override
    public IntegerTensor getLessThanOrEqualToMask(IntegerTensor lessThanOrEqualsThis) {
        if (lessThanOrEqualsThis.isLengthOne()) {
            return new ScalarIntegerTensor(value <= lessThanOrEqualsThis.scalar() ? 1 : 0);
        } else {
            return IntegerTensor.create(value, lessThanOrEqualsThis.getShape())
                .getLessThanOrEqualToMask(lessThanOrEqualsThis);
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
    public IntegerTensor setWithMask(IntegerTensor mask, Integer value) {
        return duplicate().setWithMaskInPlace(mask, value);
    }

    @Override
    public IntegerTensor apply(Function<Integer, Integer> function) {
        return duplicate().applyInPlace(function);
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
    public IntegerTensor minusInPlace(int that) {
        value = value - that;
        return this;
    }

    @Override
    public IntegerTensor plusInPlace(int that) {
        value = value + that;
        return this;
    }

    @Override
    public IntegerTensor timesInPlace(int that) {
        value = value * that;
        return this;
    }

    @Override
    public IntegerTensor divInPlace(int that) {
        value = value / that;
        return this;
    }

    @Override
    public IntegerTensor powInPlace(IntegerTensor exponent) {
        if (exponent.isLengthOne()) {
            powInPlace(exponent.scalar());
        } else {
            return IntegerTensor.create(value, exponent.getShape())
                .powInPlace(exponent);
        }
        return this;
    }

    @Override
    public IntegerTensor powInPlace(int exponent) {
        value = IntMath.pow(value, exponent);
        return this;
    }

    @Override
    public IntegerTensor minusInPlace(IntegerTensor that) {
        if (that.isLengthOne()) {
            minusInPlace(that.scalar());
        } else {
            return IntegerTensor.create(value, that.getShape())
                .minusInPlace(that);
        }
        return this;
    }

    @Override
    public IntegerTensor plusInPlace(IntegerTensor that) {
        if (that.isLengthOne()) {
            plusInPlace(that.scalar());
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
        } else {
            return IntegerTensor.create(value, that.getShape())
                .divInPlace(that);
        }
        return this;
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
    public BooleanTensor lessThan(int that) {
        return BooleanTensor.create(this.value < that);
    }

    @Override
    public BooleanTensor lessThanOrEqual(int that) {
        return BooleanTensor.create(this.value <= that);
    }

    @Override
    public BooleanTensor lessThan(IntegerTensor that) {
        if (that.isLengthOne()) {
            return lessThan(that.scalar());
        } else {
            return that.greaterThan(value);
        }
    }

    @Override
    public BooleanTensor lessThanOrEqual(IntegerTensor that) {
        if (that.isLengthOne()) {
            return lessThanOrEqual(that.scalar());
        } else {
            return that.greaterThanOrEqual(value);
        }
    }

    @Override
    public BooleanTensor greaterThan(int value) {
        return BooleanTensor.create(this.value > value);
    }

    @Override
    public BooleanTensor greaterThanOrEqual(int value) {
        return BooleanTensor.create(this.value >= value);
    }

    @Override
    public IntegerTensor minInPlace(IntegerTensor min) {
        if (min.isLengthOne()) {
            return new ScalarIntegerTensor(Math.min(value, min.scalar()));
        } else {
            return min.duplicate().minInPlace(this);
        }
    }

    @Override
    public IntegerTensor maxInPlace(IntegerTensor max) {
        if (max.isLengthOne()) {
            return new ScalarIntegerTensor(Math.max(value, max.scalar()));
        } else {
            return max.duplicate().maxInPlace(this);
        }
    }

    @Override
    public int min() {
        return value;
    }

    @Override
    public int max() {
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
    public BooleanTensor greaterThan(IntegerTensor that) {
        if (that.isLengthOne()) {
            return greaterThan(that.scalar());
        } else {
            return that.lessThan(value);
        }
    }

    @Override
    public BooleanTensor greaterThanOrEqual(IntegerTensor that) {
        if (that.isLengthOne()) {
            return greaterThanOrEqual(that.scalar());
        } else {
            return that.lessThanOrEqual(value);
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Tensor)) return false;

        Tensor that = (Tensor) o;

        return that.scalar().equals(value);
    }

    @Override
    public int hashCode() {
        int result = value != null ? value.hashCode() : 0;
        return result;
    }

    @Override
    public FlattenedView<Integer> getFlattenedView() {
        return new SimpleIntegerFlattenedView(value);
    }

    @Override
    public BooleanTensor elementwiseEquals(Tensor that) {
        if (that instanceof IntegerTensor) {
            IntegerTensor thatAsInteger = (IntegerTensor) that;
            if (that.isLengthOne()) {
                return BooleanTensor.create(value.equals(thatAsInteger.scalar()));
            } else {
                return thatAsInteger.elementwiseEquals(value);
            }
        } else {
            return Tensor.elementwiseEquals(this, that);
        }
    }

    @Override
    public BooleanTensor elementwiseEquals(Integer value) {
        return BooleanTensor.create(this.scalar().equals(value));
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

    private static class SimpleIntegerFlattenedView implements FlattenedView<Integer> {

        private int value;

        public SimpleIntegerFlattenedView(int value) {
            this.value = value;
        }

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
            this.value = value;
        }

    }

}
