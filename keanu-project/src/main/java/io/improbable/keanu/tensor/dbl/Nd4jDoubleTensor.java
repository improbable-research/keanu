package io.improbable.keanu.tensor.dbl;

import io.improbable.keanu.tensor.INDArrayExtensions;
import io.improbable.keanu.tensor.INDArrayShim;
import io.improbable.keanu.tensor.Nd4jFloatingPointTensor;
import io.improbable.keanu.tensor.Nd4jTensor;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.TypedINDArrayFactory;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.bool.JVMBooleanTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.tensor.intgr.Nd4jIntegerTensor;
import io.improbable.keanu.tensor.validate.TensorValidator;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.special.Gamma;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.scalar.LogX;
import org.nd4j.linalg.api.ops.impl.scalar.ReplaceNans;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.PowPairwise;
import org.nd4j.linalg.api.ops.impl.transforms.strict.ACosh;
import org.nd4j.linalg.api.ops.impl.transforms.strict.ASinh;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Expm1;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Log1p;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Tan;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.inverse.InvertMatrix;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

import static com.google.common.primitives.Ints.checkedCast;

/**
 * Class for representing n-dimensional arrays of doubles. This is
 * backed by Nd4j.
 */
public class Nd4jDoubleTensor extends Nd4jFloatingPointTensor<Double, DoubleTensor> implements DoubleTensor {

    static {
        INDArrayShim.startNewThreadForNd4j();
    }

    private static final DataType BUFFER_TYPE = DataType.DOUBLE;

    public Nd4jDoubleTensor(double[] data, long[] shape) {
        this(TypedINDArrayFactory.create(data, shape));
    }

    public Nd4jDoubleTensor(INDArray tensor) {
        super(tensor);
    }

    @Override
    protected INDArray getTensor(DoubleTensor tensor) {
        return getAsINDArray(tensor);
    }

    @Override
    protected DoubleTensor create(INDArray tensor) {
        return new Nd4jDoubleTensor(tensor);
    }

    @Override
    protected DoubleTensor set(INDArray tensor) {
        this.tensor = tensor.dataType() == DataType.DOUBLE ? tensor : tensor.castTo(DataType.DOUBLE);
        return this;
    }

    public static Nd4jDoubleTensor scalar(double scalarValue) {
        return new Nd4jDoubleTensor(Nd4j.scalar(scalarValue));
    }

    public static Nd4jDoubleTensor create(double[] values, long... shape) {
        long length = TensorShape.getLength(shape);
        if (values.length != length) {
            throw new IllegalArgumentException("Shape " + Arrays.toString(shape) + " does not match buffer size " + values.length);
        }
        return new Nd4jDoubleTensor(values, shape);
    }

    public static Nd4jDoubleTensor create(double value, long... shape) {
        return new Nd4jDoubleTensor(Nd4j.valueArrayOf(shape, value));
    }

    public static Nd4jDoubleTensor create(double[] values) {
        return create(values, values.length);
    }

    public static Nd4jDoubleTensor ones(long... shape) {
        return new Nd4jDoubleTensor(TypedINDArrayFactory.ones(shape, BUFFER_TYPE));
    }

    public static Nd4jDoubleTensor eye(long n) {
        return new Nd4jDoubleTensor(TypedINDArrayFactory.eye(n, BUFFER_TYPE));
    }

    public static Nd4jDoubleTensor zeros(long[] shape) {
        return new Nd4jDoubleTensor(TypedINDArrayFactory.zeros(shape, BUFFER_TYPE));
    }

    public static Nd4jDoubleTensor linspace(double start, double end, int numberOfPoints) {
        return new Nd4jDoubleTensor(TypedINDArrayFactory.linspace(start, end, numberOfPoints, BUFFER_TYPE));
    }

    public static Nd4jDoubleTensor arange(double start, double end) {
        return new Nd4jDoubleTensor(TypedINDArrayFactory.arange(start, end));
    }

    public static Nd4jDoubleTensor arange(double start, double end, double stepSize) {
        double stepCount = Math.ceil((end - start) / stepSize);
        INDArray arangeWithStep = TypedINDArrayFactory.arange(0, stepCount).muli(stepSize).addi(start);
        return new Nd4jDoubleTensor(arangeWithStep);
    }

    static INDArray getAsINDArray(DoubleTensor that) {

        if (that instanceof Nd4jTensor) {
            INDArray array = ((Nd4jTensor) that).getTensor();
            if (array.dataType() == DataType.DOUBLE) {
                return array;
            } else {
                return array.castTo(DataType.DOUBLE);
            }
        } else {
            return TypedINDArrayFactory.create(that.asFlatDoubleArray(), that.getShape());
        }
    }

    @Override
    public DoubleTensor sum(int... overDimensions) {
        if (overDimensions.length == 0) {
            return duplicate();
        }
        return new Nd4jDoubleTensor(tensor.sum(overDimensions));
    }

    @Override
    public DoubleTensor cumSumInPlace(int requestedDimension) {
        int dimension = requestedDimension >= 0 ? requestedDimension : requestedDimension + tensor.rank();
        TensorShapeValidation.checkDimensionExistsInShape(dimension, tensor.shape());
        tensor.cumsumi(dimension);
        return this;
    }

    public Double sum() {
        return tensor.sumNumber().doubleValue();
    }


    @Override
    public DoubleTensor apply(Function<Double, Double> function) {
        DataBuffer data = tensor.data().dup();
        for (int i = 0; i < data.length(); i++) {
            data.put(i, function.apply(data.getDouble(i)));
        }
        return new Nd4jDoubleTensor(data.asDouble(), this.getShape());
    }

    @Override
    public DoubleTensor matrixInverse() {
        return new Nd4jDoubleTensor(InvertMatrix.invert(tensor, false));
    }

    @Override
    public Double max() {
        return tensor.maxNumber().doubleValue();
    }

    @Override
    public Double min() {
        return tensor.minNumber().doubleValue();
    }

    @Override
    public int nanArgMax() {
        return tensor.argMax().getInt(0);
    }

    @Override
    public int argMax() {
        return duplicate()
            .replaceNaNInPlace(Double.MAX_VALUE)
            .nanArgMax();
    }

    @Override
    public IntegerTensor nanArgMax(int axis) {
        long[] shape = this.getShape();
        TensorShapeValidation.checkDimensionExistsInShape(axis, shape);
        INDArray max = tensor.argMax(axis).reshape(TensorShape.removeDimension(axis, shape));
        return new Nd4jIntegerTensor(max);
    }

    @Override
    public IntegerTensor argMax(int axis) {
        return duplicate()
            .replaceNaNInPlace(Double.MAX_VALUE)
            .nanArgMax(axis);
    }

    @Override
    public int nanArgMin() {
        return Nd4j.argMin(tensor).getInt(0);
    }

    @Override
    public int argMin() {
        return duplicate()
            .replaceNaNInPlace(-Double.MAX_VALUE)
            .nanArgMin();
    }

    @Override
    public IntegerTensor nanArgMin(int axis) {
        long[] shape = this.getShape();
        TensorShapeValidation.checkDimensionExistsInShape(axis, shape);
        return new Nd4jIntegerTensor(Nd4j.argMin(tensor, axis).reshape(TensorShape.removeDimension(axis, shape)));
    }

    @Override
    public IntegerTensor argMin(int axis) {
        return duplicate()
            .replaceNaNInPlace(-Double.MAX_VALUE)
            .nanArgMin(axis);
    }

    @Override
    public Double average() {
        return tensor.sumNumber().doubleValue() / tensor.length();
    }

    @Override
    public Double standardDeviation() {
        return tensor.stdNumber().doubleValue();
    }

    @Override
    public boolean equalsWithinEpsilon(DoubleTensor o, Double epsilon) {
        if (this == o) return true;

        if (o instanceof Nd4jTensor) {
            return tensor.equalsWithEps(((Nd4jTensor) o).getTensor(), epsilon);
        } else {
            if (this.hasSameShapeAs(o)) {
                DoubleTensor difference = o.minus(this);
                return difference.abs().lessThan(epsilon).allTrue();
            }
        }
        return false;
    }

    @Override
    public DoubleTensor choleskyDecomposition() {
        INDArray dup = tensor.dup();
        Nd4j.getBlasWrapper().lapack().potrf(dup, true);
        return new Nd4jDoubleTensor(dup);
    }

    @Override
    public Double scalar() {
        if (this.getLength() > 1) {
            throw new IllegalArgumentException("Not a scalar");
        }
        return tensor.getDouble(0);
    }

    @Override
    public DoubleTensor matrixMultiply(DoubleTensor value) {
        TensorShapeValidation.getMatrixMultiplicationResultingShape(tensor.shape(), value.getShape());
        INDArray mmulResult = tensor.mmul(getAsINDArray(value));
        return new Nd4jDoubleTensor(mmulResult);
    }

    @Override
    public DoubleTensor tensorMultiply(DoubleTensor value, int[] dimsLeft, int[] dimsRight) {
        return TensorMulByMatrixMul.tensorMmul(this, value, dimsLeft, dimsRight);
    }

    @Override
    public DoubleTensor minus(DoubleTensor that) {
        if (that.isScalar()) {
            return this.minus(that.scalar());
        } else if (this.isScalar()) {
            return that.unaryMinus().plusInPlace(this);
        } else {
            return this.duplicate().minusInPlace(that);
        }
    }

    @Override
    public DoubleTensor plus(DoubleTensor that) {
        if (that.isScalar()) {
            return this.plus(that.scalar());
        } else if (this.isScalar()) {
            return that.plus(this.scalar());
        } else {
            return this.duplicate().plusInPlace(that);
        }
    }

    @Override
    public DoubleTensor times(DoubleTensor that) {
        if (that.isScalar()) {
            return this.times(that.scalar());
        } else if (this.isScalar()) {
            return that.times(this.scalar());
        } else {
            return this.duplicate().timesInPlace(that);
        }
    }

    @Override
    public DoubleTensor div(DoubleTensor that) {
        if (that.isScalar()) {
            return this.div(that.scalar());
        } else if (this.isScalar()) {
            return that.reciprocal().timesInPlace(this);
        } else {
            return this.duplicate().divInPlace(that);
        }
    }

    @Override
    public DoubleTensor reciprocalInPlace() {
        tensor.rdivi(1.0);
        return this;
    }

    @Override
    public DoubleTensor minusInPlace(Double value) {
        tensor.subi(value);
        return this;
    }

    @Override
    public DoubleTensor plusInPlace(Double value) {
        tensor.addi(value);
        return this;
    }

    @Override
    public DoubleTensor timesInPlace(Double value) {
        tensor.muli(value);
        return this;
    }

    @Override
    public DoubleTensor divInPlace(Double value) {
        tensor.divi(value);
        return this;
    }

    @Override
    public DoubleTensor powInPlace(DoubleTensor exponent) {
        if (exponent.isScalar()) {
            Transforms.pow(tensor, exponent.scalar(), false);
        } else {
            INDArray exponentArray = getAsINDArray(exponent);
            tensor = INDArrayShim.pow(tensor, exponentArray);
        }
        return this;
    }

    @Override
    public DoubleTensor powInPlace(Double exponent) {
        Transforms.pow(tensor, exponent, false);
        return this;
    }

    @Override
    public DoubleTensor sqrtInPlace() {
        Transforms.sqrt(tensor, false);
        return this;
    }

    @Override
    public DoubleTensor logInPlace() {
        Transforms.log(tensor, false);
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
        return applyInPlace(Gamma::logGamma);
    }

    @Override
    public DoubleTensor digammaInPlace() {
        return applyInPlace(Gamma::digamma);
    }

    @Override
    public DoubleTensor sinInPlace() {
        Transforms.sin(tensor, false);
        return this;
    }

    @Override
    public DoubleTensor cosInPlace() {
        Transforms.cos(tensor, false);
        return this;
    }

    @Override
    public DoubleTensor tanInPlace() {
        Nd4j.getExecutioner().exec(new Tan(tensor, tensor));
        return this;
    }

    @Override
    public DoubleTensor atanInPlace() {
        Transforms.atan(tensor, false);
        return this;
    }

    @Override
    public DoubleTensor atan2InPlace(Double y) {
        return atan2InPlace(DoubleTensor.create(y, this.tensor.shape()));
    }

    @Override
    public DoubleTensor atan2InPlace(DoubleTensor y) {
        if (y.isScalar()) {
            tensor = Transforms.atan2(tensor, Nd4j.valueArrayOf(this.tensor.shape(), y.scalar()));
        } else {
            tensor = INDArrayShim.atan2(tensor, getAsINDArray(y));
        }
        return this;
    }

    @Override
    public DoubleTensor asinInPlace() {
        Transforms.asin(tensor, false);
        return this;
    }

    @Override
    public DoubleTensor acosInPlace() {
        Transforms.acos(tensor, false);
        return this;
    }

    @Override
    public DoubleTensor sinhInPlace() {
        Transforms.sinh(tensor, false);
        return this;
    }

    @Override
    public DoubleTensor coshInPlace() {
        Transforms.cosh(tensor, false);
        return this;
    }

    @Override
    public DoubleTensor tanhInPlace() {
        Transforms.tanh(tensor, false);
        return this;
    }

    @Override
    public DoubleTensor asinhInPlace() {
        Nd4j.getExecutioner().execAndReturn(new ASinh(tensor));
        return this;
    }

    @Override
    public DoubleTensor acoshInPlace() {
        Nd4j.getExecutioner().execAndReturn(new ACosh(tensor));
        return this;
    }

    @Override
    public DoubleTensor atanhInPlace() {
        Transforms.atanh(tensor, false);
        return this;
    }

    @Override
    public DoubleTensor expInPlace() {
        Transforms.exp(tensor, false);
        return this;
    }

    @Override
    public DoubleTensor logAddExp2InPlace(DoubleTensor that) {
        //TODO: actually use Nd4j when logsumexp is fixed in next version
        JVMDoubleTensor asJVM = JVMDoubleTensor.create(tensor.toDoubleVector(), tensor.shape());
        DoubleTensor result = asJVM.logAddExp2InPlace(that);
        return Nd4jDoubleTensor.create(result.asFlatDoubleArray(), result.getShape());
    }

    @Override
    public DoubleTensor logAddExpInPlace(DoubleTensor that) {
        //TODO: actually use Nd4j when logsumexp is fixed in next version
        JVMDoubleTensor asJVM = JVMDoubleTensor.create(tensor.toDoubleVector(), tensor.shape());
        DoubleTensor result = asJVM.logAddExpInPlace(that);
        return Nd4jDoubleTensor.create(result.asFlatDoubleArray(), result.getShape());
    }

    @Override
    public DoubleTensor log1pInPlace() {
        Nd4j.getExecutioner().exec(new Log1p(tensor));
        return this;
    }

    @Override
    public DoubleTensor log2InPlace() {
        Nd4j.getExecutioner().exec(new LogX(tensor, 2));
        return this;
    }

    @Override
    public DoubleTensor log10InPlace() {
        Nd4j.getExecutioner().exec(new LogX(tensor, 10));
        return this;
    }

    @Override
    public DoubleTensor exp2InPlace() {
        INDArray indArray = Nd4j.valueArrayOf(tensor.shape(), 2.0);
        Nd4j.getExecutioner().exec(new PowPairwise(indArray, tensor, tensor));
        return this;
    }

    @Override
    public DoubleTensor expM1InPlace() {
        Nd4j.getExecutioner().exec(new Expm1(tensor));
        return this;
    }

    /**
     * @param that Right operand.
     * @return A new DoubleTensor instance only if <i>this</i> has a length of 1 and right operand has a length greater than 1.
     * Otherwise return <i>this</i>.
     */
    @Override
    public DoubleTensor minusInPlace(DoubleTensor that) {
        if (that.isScalar()) {
            tensor.subi(that.scalar());
        } else if (this.isScalar()) {
            return this.minus(that);
        } else {
            INDArray result = INDArrayShim.subi(tensor, getAsINDArray(that).dup());
            if (result != tensor) {
                return new Nd4jDoubleTensor(result);
            }
        }
        return this;
    }

    @Override
    public DoubleTensor reverseMinusInPlace(DoubleTensor that) {
        if (that.isScalar()) {
            tensor.subi(that.scalar());
        } else if (this.isScalar()) {
            return this.minus(that);
        } else {
            INDArray result = INDArrayShim.rsubi(tensor, getAsINDArray(that).dup());
            if (result != tensor) {
                return new Nd4jDoubleTensor(result);
            }
        }
        return this;
    }

    @Override
    public DoubleTensor reverseMinusInPlace(Double value) {
        tensor.rsubi(value);
        return this;
    }

    /**
     * @param that Right operand.
     * @return A new DoubleTensor instance only if <i>this</i> has a length of 1 and right operand has a length greater than 1.
     * Otherwise return <i>this</i>.
     */
    @Override
    public DoubleTensor plusInPlace(DoubleTensor that) {
        if (that.isScalar()) {
            tensor.addi(that.scalar());
        } else if (this.isScalar()) {
            return this.plus(that);
        } else {
            INDArray result = INDArrayShim.addi(tensor, getAsINDArray(that).dup());
            if (result != tensor) {
                return new Nd4jDoubleTensor(result);
            }
        }
        return this;
    }

    /**
     * @param that Right operand.
     * @return A new DoubleTensor instance only if <i>this</i> has a length of 1 and right operand has a length greater than 1.
     * Otherwise return <i>this</i>.
     */
    @Override
    public DoubleTensor timesInPlace(DoubleTensor that) {
        if (that.isScalar()) {
            tensor.muli(that.scalar());
        } else if (this.isScalar()) {
            return this.times(that);
        } else {
            INDArray result = INDArrayShim.muli(tensor, getAsINDArray(that).dup());
            if (result != tensor) {
                return new Nd4jDoubleTensor(result);
            }
        }
        return this;
    }

    /**
     * @param that Right operand.
     * @return A new DoubleTensor instance only if <i>this</i> has a length of 1 and right operand has a length greater than 1.
     * Otherwise return <i>this</i>.
     */
    @Override
    public DoubleTensor divInPlace(DoubleTensor that) {
        if (that.isScalar()) {
            tensor.divi(that.scalar());
        } else if (this.isScalar()) {
            return this.div(that);
        } else {
            INDArray result = INDArrayShim.divi(tensor, getAsINDArray(that).dup());
            if (result != tensor) {
                return new Nd4jDoubleTensor(result);
            }
        }
        return this;
    }

    @Override
    public DoubleTensor reverseDivInPlace(Double value) {
        tensor.rdivi(value);
        return this;
    }

    @Override
    public DoubleTensor reverseDivInPlace(DoubleTensor that) {
        if (that.isScalar()) {
            tensor.subi(that.scalar());
        } else if (this.isScalar()) {
            return this.minus(that);
        } else {
            INDArray result = INDArrayShim.rdivi(tensor, getAsINDArray(that).dup());
            if (result != tensor) {
                return new Nd4jDoubleTensor(result);
            }
        }
        return this;
    }

    @Override
    public DoubleTensor unaryMinusInPlace() {
        tensor.negi();
        return this;
    }

    @Override
    public DoubleTensor absInPlace() {
        Transforms.abs(tensor, false);
        return this;
    }

    @Override
    public DoubleTensor getGreaterThanMask(DoubleTensor greaterThanThis) {
        return greaterThan(greaterThanThis).toDoubleMask();
    }

    @Override
    public DoubleTensor getGreaterThanOrEqualToMask(DoubleTensor greaterThanOrEqualToThis) {
        return greaterThanOrEqual(greaterThanOrEqualToThis).toDoubleMask();
    }

    @Override
    public BooleanTensor greaterThan(Double value) {
        return fromMask(tensor.gt(value));
    }

    @Override
    public BooleanTensor greaterThanOrEqual(Double value) {
        return fromMask(tensor.gte(value));
    }

    @Override
    public BooleanTensor greaterThan(DoubleTensor value) {

        INDArray mask;
        if (value.isScalar()) {
            mask = tensor.gt(value.scalar());
        } else {
            INDArray indArray = getAsINDArray(value);
            mask = INDArrayShim.gt(tensor, indArray);
        }

        return fromMask(mask);
    }

    @Override
    public BooleanTensor greaterThanOrEqual(DoubleTensor value) {

        INDArray mask;
        if (value.isScalar()) {
            mask = tensor.gte(value.scalar());
        } else {
            INDArray indArray = getAsINDArray(value);
            mask = tensor.dup();
            mask = INDArrayShim.gte(mask, indArray);
        }

        return fromMask(mask);
    }

    @Override
    public DoubleTensor getLessThanMask(DoubleTensor lessThanThis) {
        return lessThan(lessThanThis).toDoubleMask();
    }

    @Override
    public DoubleTensor getLessThanOrEqualToMask(DoubleTensor lessThanOrEqualToThis) {
        return lessThanOrEqual(lessThanOrEqualToThis).toDoubleMask();
    }

    @Override
    public BooleanTensor lessThan(Double value) {
        return fromMask(tensor.lt(value));
    }

    @Override
    public BooleanTensor lessThanOrEqual(Double value) {
        return fromMask(tensor.lte(value));
    }

    @Override
    public BooleanTensor lessThan(DoubleTensor value) {

        INDArray mask;
        if (value.isScalar()) {
            mask = tensor.lt(value.scalar());
        } else {
            INDArray indArray = getAsINDArray(value);
            mask = INDArrayShim.lt(tensor, indArray);
        }

        return fromMask(mask);
    }

    @Override
    public BooleanTensor lessThanOrEqual(DoubleTensor value) {

        INDArray mask;
        if (value.isScalar()) {
            mask = tensor.lte(value.scalar());
        } else {

            INDArray indArray = getAsINDArray(value);
            mask = tensor.dup();
            mask = INDArrayShim.lte(mask, indArray);
        }

        return fromMask(mask);
    }

    @Override
    public BooleanTensor elementwiseEquals(Tensor that) {
        if (that instanceof DoubleTensor) {
            if (isScalar()) {
                return (that).elementwiseEquals(this.scalar());
            } else if (that.isScalar()) {
                return elementwiseEquals(((DoubleTensor) that).scalar());
            } else {
                INDArray mask = INDArrayShim.eq(tensor, getAsINDArray((DoubleTensor) that));
                return fromMask(mask);
            }
        } else {
            return Tensor.elementwiseEquals(this, that);
        }
    }

    @Override
    public BooleanTensor elementwiseEquals(Double value) {
        return fromMask(tensor.eq(value));
    }

    private BooleanTensor fromMask(INDArray mask) {
        long[] shape = mask.shape();
        DataBuffer data = mask.data();
        boolean[] boolsFromMask = new boolean[checkedCast(mask.length())];

        for (int i = 0; i < boolsFromMask.length; i++) {
            boolsFromMask[i] = data.getDouble(i) != 0.0;
        }
        return JVMBooleanTensor.create(boolsFromMask, Arrays.copyOf(shape, shape.length));
    }

    @Override
    public DoubleTensor applyInPlace(Function<Double, Double> function) {
        DataBuffer data = tensor.data();
        for (int i = 0; i < data.length(); i++) {
            data.put(i, function.apply(data.getDouble(i)));
        }
        return this;
    }

    @Override
    public DoubleTensor maxInPlace(DoubleTensor max) {
        if (max.isScalar()) {
            Transforms.max(tensor, max.scalar(), false);
        } else {
            tensor = INDArrayShim.max(tensor, getAsINDArray(max));
        }
        return this;
    }

    @Override
    public DoubleTensor minInPlace(DoubleTensor min) {
        if (min.isScalar()) {
            Transforms.min(tensor, min.scalar(), false);
        } else {
            tensor = INDArrayShim.min(tensor, getAsINDArray(min));
        }
        return this;
    }

    @Override
    public DoubleTensor standardizeInPlace() {
        tensor.subi(average()).divi(standardDeviation());
        return this;
    }

    @Override
    public DoubleTensor setAllInPlace(Double value) {
        this.tensor.assign(value);
        return this;
    }

    @Override
    public DoubleTensor clampInPlace(DoubleTensor min, DoubleTensor max) {
        return minInPlace(max).maxInPlace(min);
    }

    public DoubleTensor ceilInPlace() {
        Transforms.ceil(tensor, false);
        return this;
    }

    @Override
    public DoubleTensor floorInPlace() {
        Transforms.floor(tensor, false);
        return this;
    }

    @Override
    public DoubleTensor roundInPlace() {
        Transforms.round(tensor, false);
        return this;
    }

    @Override
    public DoubleTensor sigmoidInPlace() {
        Transforms.sigmoid(tensor, false);
        return this;
    }

    @Override
    public Double determinant() {
        INDArray dup = tensor.dup();
        double[][] asMatrix = dup.toDoubleMatrix();
        RealMatrix matrix = new Array2DRowRealMatrix(asMatrix);
        return new LUDecomposition(matrix).getDeterminant();
    }

    @Override
    public Double product() {
        return tensor.prod().getDouble(0);
    }

    @Override
    public DoubleTensor slice(int dimension, long index) {
        return new Nd4jDoubleTensor(tensor.slice(index, dimension));
    }

    @Override
    public DoubleTensor take(long... index) {
        return scalar(getValue(index));
    }

    @Override
    public List<DoubleTensor> split(int dimension, long... splitAtIndices) {

        List<INDArray> splitINDArrays = INDArrayExtensions.split(tensor, dimension, splitAtIndices);

        return splitINDArrays.stream()
            .map(Nd4jDoubleTensor::new)
            .collect(Collectors.toList());
    }

    @Override
    public BooleanTensor notNaN() {
        return this.elementwiseEquals(this);
    }

    @Override
    public DoubleTensor replaceNaNInPlace(Double value) {
        Nd4j.getExecutioner().exec(new ReplaceNans(tensor, value));
        return this;
    }

    @Override
    public DoubleTensor toDouble() {
        return duplicate();
    }

    @Override
    public IntegerTensor toInteger() {
        return new Nd4jIntegerTensor(INDArrayExtensions.castToInteger(tensor, true));
    }

    @Override
    public double[] asFlatDoubleArray() {
        return tensor.dup().data().asDouble();
    }

    @Override
    public int[] asFlatIntegerArray() {
        return tensor.dup().data().asInt();
    }

    @Override
    public Double[] asFlatArray() {
        return ArrayUtils.toObject(asFlatDoubleArray());
    }

    @Override
    public FlattenedView<Double> getFlattenedView() {
        return new Nd4jDoubleFlattenedView();
    }

    private class Nd4jDoubleFlattenedView implements FlattenedView<Double> {

        @Override
        public long size() {
            return tensor.data().length();
        }

        @Override
        public Double get(long index) {
            return tensor.data().getDouble(index);
        }

        @Override
        public Double getOrScalar(long index) {
            if (tensor.length() == 1) {
                return get(0);
            } else {
                return get(index);
            }
        }

        @Override
        public void set(long index, Double value) {
            tensor.data().put(index, value);
        }
    }
}
