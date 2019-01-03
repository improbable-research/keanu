package io.improbable.keanu.tensor.dbl;

import com.google.common.base.Preconditions;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.tensor.validate.TensorValidator;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.special.Gamma;
import org.apache.commons.math3.util.FastMath;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.function.Function;

public class ScalarDoubleTensor implements DoubleTensor {

    private Double value;
    private long[] shape;

    private ScalarDoubleTensor(Double value, long[] shape) {
        this.value = value;
        this.shape = shape;
    }

    public ScalarDoubleTensor(double value) {
        this(value, SCALAR_SHAPE);
    }

    public ScalarDoubleTensor(long[] shape) {
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
    public long getLength() {
        return isShapePlaceholder() ? 0L : 1L;
    }

    @Override
    public boolean isShapePlaceholder() {
        return value == null;
    }

    @Override
    public ScalarDoubleTensor duplicate() {
        return new ScalarDoubleTensor(value, shape);
    }

    @Override
    public Double getValue(long[] index) {
        if (index.length == 1 && index[0] == 0) {
            return value;
        } else {
            throw new IndexOutOfBoundsException(ArrayUtils.toString(index) + " out of bounds on scalar");
        }
    }

    @Override
    public DoubleTensor setValue(Double value, long[] index) {
        if (index.length == 1 && index[0] == 0) {
            this.value = value;
            return this;
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
        return IntegerTensor.create(value.intValue(), shape);
    }

    @Override
    public Double scalar() {
        return value;
    }

    @Override
    public DoubleTensor reshape(long... newShape) {
        if (TensorShape.getLength(newShape) != 1) {
            throw new IllegalArgumentException("Cannot reshape scalar to non scalar");
        }

        return new ScalarDoubleTensor(value, newShape);
    }

    @Override
    public DoubleTensor permute(int... rearrange) {
        return new ScalarDoubleTensor(value, shape);
    }

    @Override
    public DoubleTensor diag() {
        return duplicate();
    }

    @Override
    public DoubleTensor transpose() {
        return duplicate();
    }

    /**
     * @param overDimensions the dimensions to sum over
     * @return a new scalar with a shape that has the sum over dimensions dropped
     */
    @Override
    public DoubleTensor sum(int... overDimensions) {
        overDimensions = TensorShape.getAbsoluteDimensions(shape.length, overDimensions);
        long[] summedShape = ArrayUtils.removeAll(shape, overDimensions);
        return new ScalarDoubleTensor(value, summedShape);
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
    public DoubleTensor matrixMultiply(DoubleTensor value) {
        if (value.isLengthOne()) {
            return value.times(value);
        }
        throw new IllegalArgumentException("Cannot use matrix multiply with scalar. Use times instead.");
    }

    @Override
    public DoubleTensor tensorMultiply(DoubleTensor value, int[] dimsLeft, int[] dimsRight) {
        if (value.isLengthOne()) {
            if (dimsLeft.length > 1 || dimsRight.length > 1 || dimsLeft[0] != 0 || dimsRight[0] != 0) {
                throw new IllegalArgumentException("Tensor multiply sum dimensions out of bounds for scalar");
            }
            return value.times(value);
        }
        throw new IllegalArgumentException("Cannot use tensor multiply with scalar. Use times instead.");
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
    public DoubleTensor safeLogTimes(DoubleTensor y) {
        return this.duplicate().safeLogTimesInPlace(y);
    }

    @Override
    public DoubleTensor logGamma() {
        return duplicate().logGammaInPlace();
    }

    @Override
    public DoubleTensor digamma() {
        return duplicate().digammaInPlace();
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
        if (greaterThanThis.isLengthOne()) {
            return new ScalarDoubleTensor(value > greaterThanThis.scalar() ? 1.0 : 0.0, shape);
        } else {
            return DoubleTensor.create(value, greaterThanThis.getShape())
                .getGreaterThanMask(greaterThanThis);
        }
    }

    @Override
    public DoubleTensor getGreaterThanOrEqualToMask(DoubleTensor greaterThanOrEqualToThis) {
        if (greaterThanOrEqualToThis.isLengthOne()) {
            return new ScalarDoubleTensor(value >= greaterThanOrEqualToThis.scalar() ? 1.0 : 0.0, shape);
        } else {
            return DoubleTensor.create(value, greaterThanOrEqualToThis.getShape())
                .getGreaterThanOrEqualToMask(greaterThanOrEqualToThis);
        }
    }

    @Override
    public DoubleTensor getLessThanMask(DoubleTensor lessThanThis) {
        if (lessThanThis.isLengthOne()) {
            return new ScalarDoubleTensor(value < lessThanThis.scalar() ? 1.0 : 0.0, shape);
        } else {
            return DoubleTensor.create(value, lessThanThis.getShape())
                .getLessThanMask(lessThanThis);
        }
    }

    @Override
    public DoubleTensor getLessThanOrEqualToMask(DoubleTensor lessThanOrEqualsThis) {
        if (lessThanOrEqualsThis.isLengthOne()) {
            return new ScalarDoubleTensor(value <= lessThanOrEqualsThis.scalar() ? 1.0 : 0.0, shape);
        } else {
            return DoubleTensor.create(value, lessThanOrEqualsThis.getShape())
                .getLessThanOrEqualToMask(lessThanOrEqualsThis);
        }
    }

    @Override
    public DoubleTensor setWithMaskInPlace(DoubleTensor withMask, Double valueToApply) {
        if (withMask.isLengthOne()) {
            this.value = withMask.scalar() == 1.0 ? valueToApply : this.value;
        } else {
            return DoubleTensor.create(value, withMask.getShape())
                .setWithMaskInPlace(withMask, valueToApply);
        }
        return this;
    }

    @Override
    public DoubleTensor setWithMask(DoubleTensor mask, Double value) {
        return this.duplicate().setWithMaskInPlace(mask, value);
    }

    @Override
    public DoubleTensor abs() {
        return this.duplicate().absInPlace();
    }

    @Override
    public DoubleTensor apply(Function<Double, Double> function) {
        return new ScalarDoubleTensor(function.apply(value), shape);
    }

    @Override
    public DoubleTensor matrixInverse() {
        Preconditions.checkArgument(isMatrix(), "Matrix inverse on non-matrix not allowed");
        return new ScalarDoubleTensor(1. / value, shape);
    }

    @Override
    public double max() {
        return value;
    }

    @Override
    public double min() {
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
    public double average() {
        return value;
    }

    @Override
    public double standardDeviation() {
        throw new IllegalStateException("Cannot find the standard deviation of a scalar");
    }

    @Override
    public boolean equalsWithinEpsilon(DoubleTensor o, double epsilon) {
        if (this == o) return true;
        if (!this.hasSameShapeAs(o)) return false;
        double difference = value - o.scalar();
        return (Math.abs(difference) <= epsilon);
    }

    @Override
    public DoubleTensor standardize() {
        return duplicate().standardizeInPlace();
    }

    @Override
    public DoubleTensor replaceNaN(double value) {
        return duplicate().replaceNaNInPlace(value);
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
    public DoubleTensor round() {
        return duplicate().roundInPlace();
    }

    @Override
    public DoubleTensor sigmoid() {
        return duplicate().sigmoidInPlace();
    }

    @Override
    public DoubleTensor choleskyDecomposition() {
        return duplicate();
    }

    @Override
    public double determinant() {
        return value;
    }

    @Override
    public double product() {
        return value;
    }

    @Override
    public DoubleTensor slice(int dimension, long index) {
        if (dimension == 0 && index == 0) {
            return duplicate();
        } else {
            throw new IllegalStateException("Slice is only valid for dimension and index zero in a scalar");
        }
    }

    @Override
    public List<DoubleTensor> split(int dimension, long[] splitAtIndices) {
        return Collections.singletonList(this);
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
        if (exponent.isLengthOne()) {
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
        return powInPlace(0.5);
    }

    @Override
    public DoubleTensor logInPlace() {
        value = Math.log(value);
        return this;
    }

    /**
     * This is identical to log().times(y), except that it changes NaN results to 0.
     * This is important when calculating 0log0, which should return 0
     * See https://arcsecond.wordpress.com/2009/03/19/0log0-0-for-real/ for some mathematical justification
     *
     * @param y The tensor value to multiply by
     * @return the log of this tensor multiplied by y
     */
    @Override
    public DoubleTensor safeLogTimesInPlace(DoubleTensor y) {
        TensorValidator.NAN_CATCHER.validate(this);
        TensorValidator.NAN_CATCHER.validate(y);
        DoubleTensor result = this.logInPlace().timesInPlace(y);
        return TensorValidator.NAN_FIXER.validate(result);
    }

    @Override
    public DoubleTensor logGammaInPlace() {
        value = Gamma.logGamma(value);
        return this;
    }

    @Override
    public DoubleTensor digammaInPlace() {
        value = Gamma.digamma(value);
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
        if (y.isLengthOne()) {
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
        if (that.isLengthOne()) {
            minusInPlace(that.scalar());
        } else {
            return that.unaryMinus().plusInPlace(value);
        }
        return this;
    }

    @Override
    public DoubleTensor plusInPlace(DoubleTensor that) {
        if (that.isLengthOne()) {
            plusInPlace(that.scalar());
        } else {
            return that.plus(value);
        }
        return this;
    }

    @Override
    public DoubleTensor timesInPlace(DoubleTensor that) {
        if (that.isLengthOne()) {
            timesInPlace(that.scalar());
        } else {
            return that.times(value);
        }
        return this;
    }

    @Override
    public DoubleTensor divInPlace(DoubleTensor that) {
        if (that.isLengthOne()) {
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

    public DoubleTensor maxInPlace(DoubleTensor max) {
        if (max.isLengthOne()) {
            value = Math.max(value, max.scalar());
        } else {
            return max.duplicate().maxInPlace(this);
        }
        return this;
    }

    @Override
    public DoubleTensor minInPlace(DoubleTensor min) {
        if (min.isLengthOne()) {
            value = Math.min(value, min.scalar());
        } else {
            return min.duplicate().minInPlace(this);
        }
        return this;
    }

    @Override
    public DoubleTensor clampInPlace(DoubleTensor min, DoubleTensor max) {
        return minInPlace(max).maxInPlace(min);
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

    /**
     * Note that we have modified the native Java behaviour to match Python (and therefore ND4J) behaviour
     * Which rounds negative numbers down if they end in 0.5
     * e.g.
     * Java: round(-2.5) == -2.0
     * Python: round(-2.5) == -3.0
     *
     * @return Nearest integer value as a DoubleTensor
     */
    @Override
    public DoubleTensor roundInPlace() {
        double valueToRound = value;
        if (value < 0. && value + 0.5 == (double) value.intValue()) {
            valueToRound -= 1.;
        }
        value = (double) Math.round(valueToRound);
        return this;
    }

    @Override
    public DoubleTensor sigmoidInPlace() {
        value = 1.0D / (1.0D + FastMath.exp(-value));
        return this;
    }

    @Override
    public DoubleTensor standardizeInPlace() {
        throw new IllegalStateException("Cannot standardize a scalar");
    }

    @Override
    public DoubleTensor replaceNaNInPlace(double newValue) {
        if (Double.isNaN(this.value)) {
            this.value = newValue;
        }
        return this;
    }

    @Override
    public DoubleTensor setAllInPlace(double value) {
        this.value = value;
        return this;
    }

    @Override
    public BooleanTensor lessThan(double that) {
        return BooleanTensor.create(this.value < that, shape);
    }

    @Override
    public BooleanTensor lessThanOrEqual(double that) {
        return BooleanTensor.create(this.value <= that, shape);
    }

    @Override
    public BooleanTensor lessThan(DoubleTensor that) {
        if (that.isLengthOne()) {
            return lessThan(that.scalar());
        } else {
            return that.greaterThan(value);
        }
    }

    @Override
    public BooleanTensor lessThanOrEqual(DoubleTensor that) {
        if (that.isLengthOne()) {
            return lessThanOrEqual(that.scalar());
        } else {
            return that.greaterThanOrEqual(value);
        }
    }

    @Override
    public BooleanTensor greaterThan(double value) {
        return BooleanTensor.create(this.value > value, shape);
    }

    @Override
    public BooleanTensor greaterThanOrEqual(double value) {
        return BooleanTensor.create(this.value >= value, shape);
    }

    @Override
    public BooleanTensor notNaN() {
        return BooleanTensor.create(!Double.isNaN(value), shape);
    }

    @Override
    public BooleanTensor greaterThan(DoubleTensor that) {
        if (that.isLengthOne()) {
            return greaterThan(that.scalar());
        } else {
            return that.lessThan(value);
        }
    }

    @Override
    public BooleanTensor greaterThanOrEqual(DoubleTensor that) {
        if (that.isLengthOne()) {
            return greaterThanOrEqual(that.scalar());
        } else {
            return that.lessThanOrEqual(value);
        }
    }

    @Override
    public FlattenedView<Double> getFlattenedView() {
        return new SimpleDoubleFlattenedView();
    }

    @Override
    public BooleanTensor elementwiseEquals(Tensor that) {
        if (that instanceof DoubleTensor) {
            DoubleTensor thatAsDouble = (DoubleTensor) that;
            if (that.isLengthOne()) {
                return BooleanTensor.create(value.equals(thatAsDouble.scalar()), shape);
            } else {
                return thatAsDouble.elementwiseEquals(value);
            }
        } else {
            return Tensor.elementwiseEquals(this, that);
        }
    }

    @Override
    public BooleanTensor elementwiseEquals(Double value) {
        return BooleanTensor.create(this.scalar().equals(value), shape);
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

    @Override
    public String toString() {
        return "{\n" +
            "data = [" + value + "]" +
            "\nshape = " + Arrays.toString(shape) +
            "\n}";
    }

    private class SimpleDoubleFlattenedView implements FlattenedView<Double> {


        public SimpleDoubleFlattenedView() { }

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
            ScalarDoubleTensor.this.value = value;
        }
    }
}
