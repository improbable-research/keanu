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
import java.util.Objects;
import java.util.function.Function;

import static io.improbable.keanu.tensor.TensorShape.calculateShapeForLengthOneBroadcast;
import static io.improbable.keanu.tensor.TensorShapeValidation.getTensorMultiplyResultShape;
import static org.apache.commons.lang3.ArrayUtils.removeAll;


public class ScalarDoubleTensor extends DoubleTensor {

    private double value;
    private long[] shape;

    public ScalarDoubleTensor(double value, long[] shape) {
        this.value = value;
        this.shape = shape;
    }

    public ScalarDoubleTensor(double value) {
        this(value, SCALAR_SHAPE);
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
    public void setValue(Double value, long[] index) {
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
        return IntegerTensor.create((int) value, shape);
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
    public DoubleTensor broadcast(long... toShape) {
        return DoubleTensor.create(value, toShape);
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
        long[] summedShape = removeAll(shape, overDimensions);
        return new ScalarDoubleTensor(value, summedShape);
    }

    @Override
    public DoubleTensor cumSumInPlace(int dimension) {
        return this;
    }

    @Override
    public DoubleTensor matrixMultiply(DoubleTensor that) {
        TensorShapeValidation.getMatrixMultiplicationResultingShape(shape, that.getShape());
        return that.times(value);
    }

    @Override
    public DoubleTensor tensorMultiply(DoubleTensor that, int[] dimsLeft, int[] dimsRight) {
        long[] resultShape = getTensorMultiplyResultShape(this.shape, that.getShape(), dimsLeft, dimsRight);
        return that.times(value).reshape(resultShape);
    }

    @Override
    public DoubleTensor getGreaterThanMask(DoubleTensor greaterThanThis) {
        if (greaterThanThis.isLengthOne()) {
            long[] newShape = calculateShapeForLengthOneBroadcast(shape, greaterThanThis.getShape());
            return new ScalarDoubleTensor(value > greaterThanThis.scalar() ? 1.0 : 0.0, newShape);
        } else {
            return DoubleTensor.create(value, greaterThanThis.getShape())
                .getGreaterThanMask(greaterThanThis);
        }
    }

    @Override
    public DoubleTensor getGreaterThanOrEqualToMask(DoubleTensor greaterThanOrEqualToThis) {
        if (greaterThanOrEqualToThis.isLengthOne()) {
            long[] newShape = calculateShapeForLengthOneBroadcast(shape, greaterThanOrEqualToThis.getShape());
            return new ScalarDoubleTensor(value >= greaterThanOrEqualToThis.scalar() ? 1.0 : 0.0, newShape);
        } else {
            return DoubleTensor.create(value, greaterThanOrEqualToThis.getShape())
                .getGreaterThanOrEqualToMask(greaterThanOrEqualToThis);
        }
    }

    @Override
    public DoubleTensor getLessThanMask(DoubleTensor lessThanThis) {
        if (lessThanThis.isLengthOne()) {
            long[] newShape = calculateShapeForLengthOneBroadcast(shape, lessThanThis.getShape());
            return new ScalarDoubleTensor(value < lessThanThis.scalar() ? 1.0 : 0.0, newShape);
        } else {
            return DoubleTensor.create(value, lessThanThis.getShape())
                .getLessThanMask(lessThanThis);
        }
    }

    @Override
    public DoubleTensor getLessThanOrEqualToMask(DoubleTensor lessThanOrEqualsThis) {
        if (lessThanOrEqualsThis.isLengthOne()) {
            long[] newShape = calculateShapeForLengthOneBroadcast(shape, lessThanOrEqualsThis.getShape());
            return new ScalarDoubleTensor(value <= lessThanOrEqualsThis.scalar() ? 1.0 : 0.0, newShape);
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
    public DoubleTensor matrixInverse() {
        Preconditions.checkArgument(isMatrix(), "Matrix inverse on non-matrix not allowed");
        return new ScalarDoubleTensor(1. / value, shape);
    }

    @Override
    public Double max() {
        return value;
    }

    @Override
    public Double min() {
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
    public Double average() {
        return value;
    }

    @Override
    public Double standardDeviation() {
        throw new IllegalStateException("Cannot find the standard deviation of a scalar");
    }

    @Override
    public boolean equalsWithinEpsilon(DoubleTensor o, Double epsilon) {
        if (this == o) return true;
        if (!this.hasSameShapeAs(o)) return false;
        double difference = value - o.scalar();
        return (Math.abs(difference) <= epsilon);
    }

    @Override
    public DoubleTensor choleskyDecomposition() {
        return duplicate();
    }

    @Override
    public Double determinant() {
        return value;
    }

    @Override
    public Double product() {
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
    public DoubleTensor take(long... index) {
        return new ScalarDoubleTensor(getValue(index));
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
    public DoubleTensor minusInPlace(Double that) {
        value = value - that;
        return this;
    }

    @Override
    public DoubleTensor plusInPlace(Double that) {
        value = value + that;
        return this;
    }

    @Override
    public DoubleTensor timesInPlace(Double that) {
        value = value * that;
        return this;
    }

    @Override
    public DoubleTensor divInPlace(Double that) {
        value = value / that;
        return this;
    }

    @Override
    public DoubleTensor powInPlace(DoubleTensor exponent) {
        if (exponent.isLengthOne()) {
            value = Math.pow(value, exponent.scalar());
            shape = calculateShapeForLengthOneBroadcast(shape, exponent.getShape());
        } else {
            return DoubleTensor.create(value, exponent.getShape()).powInPlace(exponent);
        }
        return this;
    }

    @Override
    public DoubleTensor powInPlace(Double exponent) {
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
    public DoubleTensor atan2InPlace(Double y) {
        value = Math.atan2(y, value);
        return this;
    }

    @Override
    public DoubleTensor atan2InPlace(DoubleTensor y) {
        if (y.isLengthOne()) {
            value = Math.atan2(y.scalar(), value);
            shape = calculateShapeForLengthOneBroadcast(shape, y.getShape());
        } else {
            return DoubleTensor.create(value, y.getShape()).atan2InPlace(y);
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
            shape = calculateShapeForLengthOneBroadcast(shape, that.getShape());
        } else {
            return that.reverseMinus(value);
        }
        return this;
    }

    @Override
    public DoubleTensor reverseMinusInPlace(DoubleTensor that) {
        if (that.isLengthOne()) {
            reverseMinusInPlace(that.scalar());
            shape = calculateShapeForLengthOneBroadcast(shape, that.getShape());
        } else {
            return that.minus(value);
        }
        return this;
    }

    @Override
    public DoubleTensor reverseMinusInPlace(Double that) {
        value = that - value;
        return this;
    }

    @Override
    public DoubleTensor plusInPlace(DoubleTensor that) {
        if (that.isLengthOne()) {
            plusInPlace(that.scalar());
            shape = calculateShapeForLengthOneBroadcast(shape, that.getShape());
        } else {
            return that.plus(value);
        }
        return this;
    }

    @Override
    public DoubleTensor timesInPlace(DoubleTensor that) {
        if (that.isLengthOne()) {
            timesInPlace(that.scalar());
            shape = calculateShapeForLengthOneBroadcast(shape, that.getShape());
        } else {
            return that.times(value);
        }
        return this;
    }

    @Override
    public DoubleTensor divInPlace(DoubleTensor that) {
        if (that.isLengthOne()) {
            divInPlace(that.scalar());
            shape = calculateShapeForLengthOneBroadcast(shape, that.getShape());
        } else {
            return that.reverseDiv(value);
        }
        return this;
    }

    @Override
    public DoubleTensor reverseDivInPlace(Double that) {
        value = that - value;
        return this;
    }

    @Override
    public DoubleTensor reverseDivInPlace(DoubleTensor that) {
        if (that.isLengthOne()) {
            reverseDivInPlace(that.scalar());
            shape = calculateShapeForLengthOneBroadcast(shape, that.getShape());
        } else {
            return that.div(value);
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
            shape = calculateShapeForLengthOneBroadcast(shape, max.getShape());
        } else {
            return max.duplicate().maxInPlace(this);
        }
        return this;
    }

    @Override
    public DoubleTensor minInPlace(DoubleTensor min) {
        if (min.isLengthOne()) {
            value = Math.min(value, min.scalar());
            shape = calculateShapeForLengthOneBroadcast(shape, min.getShape());
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

    @Override
    public DoubleTensor roundInPlace() {
        double valueToRound = value;
        if (value < 0. && value + 0.5 == value) {
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
    public DoubleTensor replaceNaNInPlace(Double newValue) {
        if (Double.isNaN(this.value)) {
            this.value = newValue;
        }
        return this;
    }

    @Override
    public DoubleTensor setAllInPlace(Double value) {
        this.value = value;
        return this;
    }

    @Override
    public BooleanTensor lessThan(Double that) {
        return BooleanTensor.create(this.value < that, shape);
    }

    @Override
    public BooleanTensor lessThanOrEqual(Double that) {
        return BooleanTensor.create(this.value <= that, shape);
    }

    @Override
    public BooleanTensor lessThan(DoubleTensor that) {
        if (that.isLengthOne()) {
            long[] newShape = calculateShapeForLengthOneBroadcast(shape, that.getShape());
            return BooleanTensor.create(this.value < that.scalar(), newShape);
        } else {
            return that.greaterThan(value);
        }
    }

    @Override
    public BooleanTensor lessThanOrEqual(DoubleTensor that) {
        if (that.isLengthOne()) {
            long[] newShape = calculateShapeForLengthOneBroadcast(shape, that.getShape());
            return BooleanTensor.create(this.value <= that.scalar(), newShape);
        } else {
            return that.greaterThanOrEqual(value);
        }
    }

    @Override
    public BooleanTensor greaterThan(Double value) {
        return BooleanTensor.create(this.value > value, shape);
    }

    @Override
    public BooleanTensor greaterThanOrEqual(Double value) {
        return BooleanTensor.create(this.value >= value, shape);
    }

    @Override
    public BooleanTensor notNaN() {
        return BooleanTensor.create(!Double.isNaN(value), shape);
    }

    @Override
    public BooleanTensor greaterThan(DoubleTensor that) {
        if (that.isLengthOne()) {
            long[] newShape = calculateShapeForLengthOneBroadcast(shape, that.getShape());
            return BooleanTensor.create(this.value > that.scalar(), newShape);
        } else {
            return that.lessThan(value);
        }
    }

    @Override
    public BooleanTensor greaterThanOrEqual(DoubleTensor that) {
        if (that.isLengthOne()) {
            long[] newShape = calculateShapeForLengthOneBroadcast(shape, that.getShape());
            return BooleanTensor.create(this.value >= that.scalar(), newShape);
        } else {
            return that.lessThanOrEqual(value);
        }
    }

    @Override
    public FlattenedView<Double> getFlattenedView() {
        return new ScalarDoubleFlattenedView();
    }

    @Override
    public BooleanTensor elementwiseEquals(Tensor that) {
        if (that instanceof DoubleTensor) {
            DoubleTensor thatAsDouble = (DoubleTensor) that;
            if (that.isLengthOne()) {
                long[] newShape = calculateShapeForLengthOneBroadcast(shape, that.getShape());
                return BooleanTensor.create(new Double(value).equals(thatAsDouble.scalar()), newShape);
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
        return new int[]{(int) value};
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
        int result = Objects.hash(value);
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

    private class ScalarDoubleFlattenedView implements FlattenedView<Double> {

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
