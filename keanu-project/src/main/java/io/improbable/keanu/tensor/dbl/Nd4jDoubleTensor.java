package io.improbable.keanu.tensor.dbl;

import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.tensor.intgr.Nd4jIntegerTensor;
import io.improbable.keanu.tensor.ndj4.INDArrayShim;
import io.improbable.keanu.tensor.ndj4.Nd4jFloatingPointTensor;
import io.improbable.keanu.tensor.ndj4.Nd4jTensor;
import io.improbable.keanu.tensor.ndj4.TypedINDArrayFactory;
import io.improbable.keanu.tensor.validate.TensorValidator;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.special.Gamma;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.scalar.ReplaceNans;
import org.nd4j.linalg.api.ops.impl.transforms.bool.MatchConditionTransform;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Conditions;

import java.util.Arrays;

import static io.improbable.keanu.tensor.TensorShape.getBroadcastResultShape;
import static io.improbable.keanu.tensor.ndj4.INDArrayExtensions.asBoolean;
import static java.util.Arrays.copyOf;

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

    public Nd4jDoubleTensor(DoubleTensor from) {
        this(TypedINDArrayFactory.create(from.asFlatDoubleArray(), from.getShape()));
    }

    @Override
    protected INDArray getTensor(Tensor tensor) {
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

    @Override
    protected DoubleTensor getThis() {
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

    static INDArray getAsINDArray(Tensor that) {

        if (that instanceof Nd4jTensor) {
            INDArray array = ((Nd4jTensor) that).getTensor();
            if (array.dataType() == DataType.DOUBLE) {
                return array;
            } else {
                return array.castTo(DataType.DOUBLE);
            }
        } else if (that instanceof NumberTensor) {
            return TypedINDArrayFactory.create(((NumberTensor) that).toDouble().asFlatDoubleArray(), that.getShape());
        } else {
            throw new IllegalArgumentException("Cannot convert " + that.getClass().getSimpleName() + " to double INDArray/");
        }
    }

    /**
     * Nd4j DiagPart doesnt support non-square matrix diag. In the case where this is non-square
     * the JVMDoubleTensor implementation is used. For square matrices, the nd4j implementation is used.
     *
     * @return matrices with their diagonals equal to the batched vectors from this.
     */
    @Override
    public DoubleTensor diag() {
        if (getRank() > 1) {
            return new Nd4jDoubleTensor(JVMDoubleTensor.create(asFlatDoubleArray(), getShape()).diag());
        } else {
            return super.diag();
        }
    }

    /**
     * Nd4j DiagPart doesnt support non-square matrix diag. In the case where this is non-square
     * the JVMDoubleTensor implementation is used. For square matrices, the nd4j implementation is used.
     *
     * @return The elements from the diagonal of this matrix.
     */
    @Override
    public DoubleTensor diagPart() {
        if (tensor.size(0) != tensor.size(1)) {
            return new Nd4jDoubleTensor(JVMDoubleTensor.create(asFlatDoubleArray(), getShape()).diagPart());
        } else {
            return super.diagPart();
        }
    }

    @Override
    public IntegerTensor nanArgMax() {
        return IntegerTensor.scalar(tensor.argMax().getInt(0));
    }

    @Override
    public IntegerTensor argMax() {
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
    public IntegerTensor nanArgMin() {
        return IntegerTensor.scalar(Nd4j.argMin(tensor).getInt(0));
    }

    @Override
    public IntegerTensor argMin() {
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
    protected Double getNumber(Number number) {
        return number.doubleValue();
    }

    @Override
    public DoubleTensor greaterThanMask(DoubleTensor greaterThanThis) {
        return greaterThan(greaterThanThis).toDoubleMask();
    }

    @Override
    public DoubleTensor greaterThanOrEqualToMask(DoubleTensor greaterThanOrEqualToThis) {
        return greaterThanOrEqual(greaterThanOrEqualToThis).toDoubleMask();
    }

    @Override
    public DoubleTensor lessThanMask(DoubleTensor lessThanThis) {
        return lessThan(lessThanThis).toDoubleMask();
    }

    @Override
    public DoubleTensor lessThanOrEqualToMask(DoubleTensor lessThanOrEqualToThis) {
        return lessThanOrEqual(lessThanOrEqualToThis).toDoubleMask();
    }

    @Override
    public DoubleTensor safeLogTimesInPlace(DoubleTensor y) {
        TensorValidator.NAN_CATCHER.validate(getThis());
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
    public DoubleTensor replaceNaNInPlace(Double value) {
        Nd4j.getExecutioner().exec(new ReplaceNans(tensor, value));
        return this;
    }

    @Override
    public BooleanTensor isFinite() {
        INDArray result = Nd4j.getExecutioner().exec(
            new MatchConditionTransform(
                tensor, Nd4j.createUninitialized(DataType.BOOL, tensor.shape(), tensor.ordering()),
                Conditions.isFinite())
        );
        return BooleanTensor.create(asBoolean(result), tensor.shape());
    }

    @Override
    public BooleanTensor isInfinite() {
        INDArray result = tensor.isInfinite();
        return BooleanTensor.create(asBoolean(result), tensor.shape());
    }

    @Override
    public BooleanTensor isNegativeInfinity() {
        INDArray result = Nd4j.getExecutioner().exec(
            new MatchConditionTransform(
                tensor, Nd4j.createUninitialized(DataType.BOOL, tensor.shape(), tensor.ordering()),
                Conditions.equals(Double.NEGATIVE_INFINITY))
        );
        return BooleanTensor.create(asBoolean(result), tensor.shape());
    }

    @Override
    public BooleanTensor isPositiveInfinity() {
        INDArray result = Nd4j.getExecutioner().exec(
            new MatchConditionTransform(
                tensor, Nd4j.createUninitialized(DataType.BOOL, tensor.shape(), tensor.ordering()),
                Conditions.equals(Double.POSITIVE_INFINITY))
        );
        return BooleanTensor.create(asBoolean(result), tensor.shape());
    }

    @Override
    public DoubleTensor where(BooleanTensor predicate, DoubleTensor els) {
        final long[] resultShape = getBroadcastResultShape(getBroadcastResultShape(getShape(), predicate.getShape()), els.getShape());
        final DoubleTensor broadcastedTrue = this.hasShape(resultShape) ? this : this.broadcast(resultShape);
        final DoubleTensor broadcastedFalse = els.hasShape(resultShape) ? els : els.broadcast(resultShape);
        final BooleanTensor broadcastedPredicate = predicate.hasShape(resultShape) ? predicate : predicate.broadcast(resultShape);

        FlattenedView<Double> trueValuesFlattened = broadcastedTrue.getFlattenedView();
        FlattenedView<Double> falseValuesFlattened = broadcastedFalse.getFlattenedView();
        FlattenedView<Boolean> predicateValuesFlattened = broadcastedPredicate.getFlattenedView();

        double[] result = new double[TensorShape.getLengthAsInt(resultShape)];
        for (int i = 0; i < result.length; i++) {
            result[i] = predicateValuesFlattened.get(i) ? trueValuesFlattened.get(i) : falseValuesFlattened.get(i);
        }

        return DoubleTensor.create(result, copyOf(resultShape, resultShape.length));
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
