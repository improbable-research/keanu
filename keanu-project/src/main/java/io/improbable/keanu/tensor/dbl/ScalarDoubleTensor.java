package io.improbable.keanu.tensor.dbl;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.util.FastMath;

import java.util.Arrays;
import java.util.function.Function;

public class ScalarDoubleTensor implements DoubleTensor {

    private Double value;
    private int[] shape;

    public ScalarDoubleTensor(double value) {
        this.value = value;
        this.shape = SCALAR_SHAPE;
    }

    public ScalarDoubleTensor(int[] shape) {
        this.value = null;
        this.shape = shape;
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
        return isShapePlaceholder() ? 0L : 1L;
    }

    @Override
    public boolean isShapePlaceholder() {
        return value == null;
    }

    @Override
    public DoubleTensor duplicate() {
        return new ScalarDoubleTensor(value);
    }

    @Override
    public Double getValue(int[] index) {
        if (index.length == 1 && index[0] == 0) {
            return value;
        } else {
            throw new IndexOutOfBoundsException(ArrayUtils.toString(index) + " out of bounds on scalar");
        }
    }

    @Override
    public void setValue(Double value, int[] index) {
        if (index.length == 1 && index[0] == 0) {
            this.value = value;
        } else {
            throw new IndexOutOfBoundsException(ArrayUtils.toString(index) + " out of bounds on scalar");
        }
    }

    @Override
    public Double sum() {
        return value;
    }

    @Override
    public DoubleTensor toDouble() {
        return this;
    }

    @Override
    public IntegerTensor toInteger() {
        return IntegerTensor.scalar(value.intValue());
    }

    @Override
    public Double scalar() {
        return value;
    }

    @Override
    public DoubleTensor reciprocal() {
        return this.duplicate().reciprocalInPlace();
    }

    @Override
    public DoubleTensor minus(double that) {
        return this.duplicate().minusInPlace(that);
    }

    @Override
    public DoubleTensor plus(double that) {
        return this.duplicate().plusInPlace(that);
    }

    @Override
    public DoubleTensor times(double that) {
        return this.duplicate().timesInPlace(that);
    }

    @Override
    public DoubleTensor div(double that) {
        return this.duplicate().divInPlace(that);
    }

    @Override
    public DoubleTensor pow(DoubleTensor exponent) {
        return this.duplicate().powInPlace(exponent);
    }

    @Override
    public DoubleTensor pow(double exponent) {
        return this.duplicate().powInPlace(exponent);
    }

    @Override
    public DoubleTensor sqrt() {
        return pow(0.5);
    }

    @Override
    public DoubleTensor log() {
        return this.duplicate().logInPlace();
    }

    @Override
    public DoubleTensor sin() {
        return this.duplicate().sinInPlace();
    }

    @Override
    public DoubleTensor cos() {
        return this.duplicate().cosInPlace();
    }

    @Override
    public DoubleTensor tan() {
        return this.duplicate().tanInPlace();
    }

    @Override
    public DoubleTensor atan() {
        return this.duplicate().atanInPlace();
    }

    @Override
    public DoubleTensor atan2(double y) {
        return this.duplicate().atan2InPlace(y);
    }

    @Override
    public DoubleTensor atan2(DoubleTensor y) {
        return this.duplicate().atan2InPlace(y);
    }

    @Override
    public DoubleTensor asin() {
        return this.duplicate().asinInPlace();
    }

    @Override
    public DoubleTensor acos() {
        return this.duplicate().acosInPlace();
    }

    @Override
    public DoubleTensor exp() {
        return this.duplicate().expInPlace();
    }

    @Override
    public DoubleTensor minus(DoubleTensor that) {
        return this.duplicate().minusInPlace(that);
    }

    @Override
    public DoubleTensor plus(DoubleTensor that) {
        return this.duplicate().plusInPlace(that);
    }

    @Override
    public DoubleTensor times(DoubleTensor that) {
        return this.duplicate().timesInPlace(that);
    }

    @Override
    public DoubleTensor div(DoubleTensor that) {
        return this.duplicate().divInPlace(that);
    }

    @Override
    public DoubleTensor unaryMinus() {
        return this.duplicate().unaryMinusInPlace();
    }

    @Override
    public DoubleTensor getGreaterThanMask(DoubleTensor greaterThanThis) {
        if (greaterThanThis.isScalar()) {
            return new ScalarDoubleTensor(value > greaterThanThis.scalar() ? 1 : 0);
        } else {
            return DoubleTensor.create(value, greaterThanThis.getShape())
                .getGreaterThanMask(greaterThanThis);
        }
    }

    @Override
    public DoubleTensor getGreaterThanOrEqualToMask(DoubleTensor greaterThanOrEqualToThis) {
        if (greaterThanOrEqualToThis.isScalar()) {
            return new ScalarDoubleTensor(value >= greaterThanOrEqualToThis.scalar() ? 1 : 0);
        } else {
            return DoubleTensor.create(value, greaterThanOrEqualToThis.getShape())
                .getGreaterThanOrEqualToMask(greaterThanOrEqualToThis);
        }
    }

    @Override
    public DoubleTensor getLessThanMask(DoubleTensor lessThanThis) {
        if (lessThanThis.isScalar()) {
            return new ScalarDoubleTensor(value < lessThanThis.scalar() ? 1 : 0);
        } else {
            return DoubleTensor.create(value, lessThanThis.getShape())
                .getLessThanMask(lessThanThis);
        }
    }

    @Override
    public DoubleTensor getLessThanOrEqualToMask(DoubleTensor lessThanOrEqualsThis) {
        if (lessThanOrEqualsThis.isScalar()) {
            return new ScalarDoubleTensor(value <= lessThanOrEqualsThis.scalar() ? 1 : 0);
        } else {
            return DoubleTensor.create(value, lessThanOrEqualsThis.getShape())
                .getLessThanOrEqualToMask(lessThanOrEqualsThis);
        }
    }

    @Override
    public DoubleTensor setWithMaskInPlace(DoubleTensor withMask, double valueToApply) {
        if (withMask.isScalar()) {
            this.value = withMask.scalar() == 1.0 ? valueToApply : this.value;
        } else {
            return DoubleTensor.create(value, withMask.getShape())
                .setWithMaskInPlace(withMask, valueToApply);
        }
        return this;
    }

    @Override
    public DoubleTensor setWithMask(DoubleTensor mask, double value) {
        return this.duplicate().setWithMaskInPlace(mask, value);
    }

    @Override
    public DoubleTensor abs() {
        return this.duplicate().absInPlace();
    }

    @Override
    public DoubleTensor apply(Function<Double, Double> function) {
        return new ScalarDoubleTensor(function.apply(value));
    }

    @Override
    public DoubleTensor max(DoubleTensor max) {
        return duplicate().maxInPlace(max);
    }

    @Override
    public DoubleTensor min(DoubleTensor min) {
        if (min.isScalar()) {
            return new ScalarDoubleTensor(Math.min(value, min.scalar()));
        } else {
            return DoubleTensor.create(value, shape).minInPlace(min);
        }
    }

    @Override
    public DoubleTensor clamp(DoubleTensor min, DoubleTensor max) {
        return duplicate().clampInPlace(min, max);
    }

    @Override
    public DoubleTensor ceil() {
        return duplicate().ceilInPlace();
    }

    @Override
    public DoubleTensor floor() {
        return duplicate().floorInPlace();
    }

    @Override
    public DoubleTensor sigmoid() {
        return duplicate().sigmoidInPlace();
    }

    @Override
    public DoubleTensor reciprocalInPlace() {
        value = 1.0 / value;
        return this;
    }

    @Override
    public DoubleTensor minusInPlace(double that) {
        value = value - that;
        return this;
    }

    @Override
    public DoubleTensor plusInPlace(double that) {
        value = value + that;
        return this;
    }

    @Override
    public DoubleTensor timesInPlace(double that) {
        value = value * that;
        return this;
    }

    @Override
    public DoubleTensor divInPlace(double that) {
        value = value / that;
        return this;
    }

    @Override
    public DoubleTensor powInPlace(DoubleTensor exponent) {
        if (exponent.isScalar()) {
            value = Math.pow(value, exponent.scalar());
        } else {
            return DoubleTensor.create(value, exponent.getShape()).powInPlace(exponent);
        }
        return this;
    }

    @Override
    public DoubleTensor powInPlace(double exponent) {
        value = Math.pow(value, exponent);
        return this;
    }

    @Override
    public DoubleTensor sqrtInPlace() {
        return pow(0.5);
    }

    @Override
    public DoubleTensor logInPlace() {
        value = Math.log(value);
        return this;
    }

    @Override
    public DoubleTensor sinInPlace() {
        value = Math.sin(value);
        return this;
    }

    @Override
    public DoubleTensor cosInPlace() {
        value = Math.cos(value);
        return this;
    }

    @Override
    public DoubleTensor tanInPlace() {
        value = Math.tan(value);
        return this;
    }

    @Override
    public DoubleTensor atanInPlace() {
        value = Math.atan(value);
        return this;
    }

    @Override
    public DoubleTensor atan2InPlace(double y) {
        value = Math.atan2(y, value);
        return this;
    }

    @Override
    public DoubleTensor atan2InPlace(DoubleTensor y) {
        if (y.isScalar()) {
            value = Math.atan2(y.scalar(), value);
        } else {
            return Nd4jDoubleTensor.create(value, y.getShape()).atan2InPlace(y);
        }
        return this;
    }

    @Override
    public DoubleTensor asinInPlace() {
        value = Math.asin(value);
        return this;
    }

    @Override
    public DoubleTensor acosInPlace() {
        value = Math.acos(value);
        return this;
    }

    @Override
    public DoubleTensor expInPlace() {
        value = Math.exp(value);
        return this;
    }

    @Override
    public DoubleTensor minusInPlace(DoubleTensor that) {
        if (that.isScalar()) {
            minusInPlace(that.scalar());
        } else {
            return that.unaryMinus().plusInPlace(value);
        }
        return this;
    }

    @Override
    public DoubleTensor plusInPlace(DoubleTensor that) {
        if (that.isScalar()) {
            plusInPlace(that.scalar());
        } else {
            return that.plus(value);
        }
        return this;
    }

    @Override
    public DoubleTensor timesInPlace(DoubleTensor that) {
        if (that.isScalar()) {
            timesInPlace(that.scalar());
        } else {
            return that.times(value);
        }
        return this;
    }

    @Override
    public DoubleTensor divInPlace(DoubleTensor that) {
        if (that.isScalar()) {
            divInPlace(that.scalar());
        } else {
            return that.reciprocal().timesInPlace(value);
        }
        return this;
    }

    @Override
    public DoubleTensor unaryMinusInPlace() {
        value = -value;
        return this;
    }

    @Override
    public DoubleTensor absInPlace() {
        value = Math.abs(value);
        return this;
    }

    @Override
    public DoubleTensor applyInPlace(Function<Double, Double> function) {
        return apply(function);
    }

    @Override
    public DoubleTensor maxInPlace(DoubleTensor max) {
        if (max.isScalar()) {
            value = Math.max(value, max.scalar());
        } else {
            return DoubleTensor.create(value, shape).maxInPlace(max);
        }
        return this;
    }

    @Override
    public DoubleTensor minInPlace(DoubleTensor min) {
        if (min.isScalar()) {
            value = Math.min(value, min.scalar());
        } else {
            return DoubleTensor.create(value, shape).minInPlace(min);
        }
        return this;
    }

    @Override
    public DoubleTensor clampInPlace(DoubleTensor min, DoubleTensor max) {
        return minusInPlace(min).maxInPlace(max);
    }

    @Override
    public DoubleTensor ceilInPlace() {
        value = Math.ceil(value);
        return this;
    }

    @Override
    public DoubleTensor floorInPlace() {
        value = Math.floor(value);
        return this;
    }

    @Override
    public DoubleTensor sigmoidInPlace() {
        value = 1.0D / (1.0D + FastMath.exp(-value));
        return this;
    }

    @Override
    public BooleanTensor lessThan(double that) {
        return BooleanTensor.scalar(this.value < that);
    }

    @Override
    public BooleanTensor lessThanOrEqual(double that) {
        return BooleanTensor.scalar(this.value <= that);
    }

    @Override
    public BooleanTensor lessThan(DoubleTensor that) {
        if (that.isScalar()) {
            return lessThan(that.scalar());
        } else {
            return that.greaterThan(value);
        }
    }

    @Override
    public BooleanTensor lessThanOrEqual(DoubleTensor that) {
        if (that.isScalar()) {
            return lessThanOrEqual(that.scalar());
        } else {
            return that.greaterThanOrEqual(value);
        }
    }

    @Override
    public BooleanTensor greaterThan(double value) {
        return BooleanTensor.scalar(this.value > value);
    }

    @Override
    public BooleanTensor greaterThanOrEqual(double value) {
        return BooleanTensor.scalar(this.value >= value);
    }

    @Override
    public BooleanTensor greaterThan(DoubleTensor that) {
        if (that.isScalar()) {
            return greaterThan(that.scalar());
        } else {
            return that.lessThan(value);
        }
    }

    @Override
    public BooleanTensor greaterThanOrEqual(DoubleTensor that) {
        if (that.isScalar()) {
            return greaterThanOrEqual(that.scalar());
        } else {
            return that.lessThanOrEqual(value);
        }
    }

    @Override
    public FlattenedView<Double> getFlattenedView() {
        return new SimpleDoubleFlattenedView(value);
    }

    private static class SimpleDoubleFlattenedView implements FlattenedView<Double> {

        private double value;

        public SimpleDoubleFlattenedView(double value) {
            this.value = value;
        }

        @Override
        public long size() {
            return 1;
        }

        @Override
        public Double get(long index) {
            if (index != 0) {
                throw new IndexOutOfBoundsException();
            }
            return value;
        }

        @Override
        public Double getOrScalar(long index) {
            return value;
        }

        @Override
        public void set(long index, Double value) {
            if (index != 0) {
                throw new IndexOutOfBoundsException();
            }
            this.value = value;
        }
    }

    @Override
    public double[] asFlatDoubleArray() {
        return new double[]{value};
    }

    @Override
    public int[] asFlatIntegerArray() {
        return new int[]{value.intValue()};
    }

    @Override
    public Double[] asFlatArray() {
        return ArrayUtils.toObject(asFlatDoubleArray());
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
}
