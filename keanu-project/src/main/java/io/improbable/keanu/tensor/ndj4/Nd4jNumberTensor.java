package io.improbable.keanu.tensor.ndj4;

import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.bool.JVMBooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.tensor.dbl.TensorMulByMatrixMul;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.tensor.intgr.Nd4jIntegerTensor;
import io.improbable.keanu.tensor.lng.LongTensor;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarMax;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarMin;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.CompareAndSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;
import java.util.function.Function;

import static com.google.common.primitives.Ints.checkedCast;
import static io.improbable.keanu.tensor.ndj4.INDArrayExtensions.asBoolean;

public abstract class Nd4jNumberTensor<T extends Number, TENSOR extends NumberTensor<T, TENSOR>>
    extends Nd4jTensor<T, TENSOR> implements NumberTensor<T, TENSOR> {

    public Nd4jNumberTensor(INDArray tensor) {
        super(tensor);
    }

    @Override
    public TENSOR setWithMaskInPlace(TENSOR mask, T value) {

        INDArray maskINDArray = getTensor(mask);

        //Nd4j compare and set only works for fp types
        INDArray dblBuffer = tensor.dataType() == DataType.DOUBLE ? tensor : tensor.castTo(DataType.DOUBLE);
        INDArray dblMask = maskINDArray.dataType() == DataType.DOUBLE ? maskINDArray : maskINDArray.castTo(DataType.DOUBLE);

        if (!Arrays.equals(dblBuffer.shape(), dblMask.shape())) {
            long[] broadcastShape = TensorShape.getBroadcastResultShape(dblBuffer.shape(), dblMask.shape());
            dblBuffer = dblBuffer.broadcast(broadcastShape);
            dblMask = dblMask.broadcast(broadcastShape);
        }

        double dblValue = value.doubleValue();

        double trueValue = 1.0;
        if (dblValue == 0.0) {
            trueValue = 1.0 - trueValue;
            dblMask.negi().addi(1.0);
        }
        double falseValue = 1.0 - trueValue;

        Nd4j.getExecutioner().exec(
            new CompareAndSet(dblMask, dblValue, Conditions.equals(trueValue))
        );

        Nd4j.getExecutioner().exec(
            new CompareAndSet(dblBuffer, dblMask, Conditions.notEquals(falseValue))
        );

        return set(dblBuffer);
    }

    @Override
    public T sumNumber() {
        return getNumber(tensor.sumNumber());
    }

    @Override
    public TENSOR sum(int... overDimensions) {
        if (overDimensions.length == 0) {
            return duplicate();
        }
        return create(tensor.sum(overDimensions));
    }

    @Override
    public TENSOR sum() {
        return create(Nd4j.scalar(tensor.sumNumber().doubleValue()).castTo(tensor.dataType()));
    }

    @Override
    public TENSOR cumSumInPlace(int requestedDimension) {
        int dimension = requestedDimension >= 0 ? requestedDimension : requestedDimension + tensor.rank();
        TensorShapeValidation.checkDimensionExistsInShape(dimension, tensor.shape());
        return set(tensor.cumsumi(dimension));
    }

    @Override
    public TENSOR product() {
        return create(Nd4j.scalar(tensor.prodNumber().doubleValue()).castTo(tensor.dataType()));
    }

    @Override
    public TENSOR product(int... overDimensions) {
        return create(tensor.prod(overDimensions));
    }

    @Override
    public TENSOR cumProdInPlace(int requestedDimension) {
        int dimension = requestedDimension >= 0 ? requestedDimension : requestedDimension + tensor.rank();
        TensorShapeValidation.checkDimensionExistsInShape(dimension, tensor.shape());
        return set(INDArrayExtensions.cumProd(this.tensor, dimension));
    }

    @Override
    public TENSOR clampInPlace(TENSOR min, TENSOR max) {
        return minInPlace(max).maxInPlace(min);
    }

    @Override
    public TENSOR max() {
        return create(Nd4j.scalar(tensor.maxNumber().doubleValue()).castTo(tensor.dataType()));
    }

    @Override
    public TENSOR min() {
        return create(Nd4j.scalar(tensor.minNumber().doubleValue()).castTo(tensor.dataType()));
    }

    @Override
    public TENSOR maxInPlace(TENSOR max) {
        if (max.isScalar()) {
            Nd4j.getExecutioner().exec(new ScalarMax(tensor, max.scalar()));
        } else {
            tensor = INDArrayShim.max(tensor, getTensor(max));
        }
        return set(tensor);
    }

    @Override
    public TENSOR minInPlace(TENSOR min) {
        if (min.isScalar()) {
            Nd4j.getExecutioner().exec(new ScalarMin(tensor, min.scalar()));
        } else {
            tensor = INDArrayShim.min(tensor, getTensor(min));
        }
        return set(tensor);
    }

    @Override
    public IntegerTensor argMax() {
        return IntegerTensor.scalar(tensor.argMax().getInt(0));
    }

    @Override
    public IntegerTensor argMax(int axis) {
        long[] shape = this.getShape();
        TensorShapeValidation.checkDimensionExistsInShape(axis, shape);
        INDArray max = tensor.argMax(axis).reshape(TensorShape.removeDimension(axis, shape));
        return new Nd4jIntegerTensor(max);
    }

    @Override
    public IntegerTensor argMin() {
        return IntegerTensor.scalar(Nd4j.argMin(tensor).getInt(0));
    }

    @Override
    public IntegerTensor argMin(int axis) {
        long[] shape = this.getShape();
        TensorShapeValidation.checkDimensionExistsInShape(axis, shape);
        return new Nd4jIntegerTensor(Nd4j.argMin(tensor, axis).reshape(TensorShape.removeDimension(axis, shape)));
    }

    @Override
    public TENSOR minus(TENSOR that) {
        if (that.isScalar()) {
            return this.minus(that.scalar());
        } else if (this.isScalar()) {
            return that.reverseMinus(this.scalar());
        } else {
            return this.duplicate().minusInPlace(that);
        }
    }

    @Override
    public TENSOR plus(TENSOR that) {
        if (that.isScalar()) {
            return this.plus(that.scalar());
        } else if (this.isScalar()) {
            return that.plus(this.scalar());
        } else {
            return this.duplicate().plusInPlace(that);
        }
    }

    @Override
    public TENSOR times(TENSOR that) {
        if (that.isScalar()) {
            return this.times(that.scalar());
        } else if (this.isScalar()) {
            return that.times(this.scalar());
        } else {
            return this.duplicate().timesInPlace(that);
        }
    }

    @Override
    public TENSOR div(TENSOR that) {
        if (that.isScalar()) {
            return this.div(that.scalar());
        } else if (this.isScalar()) {
            return that.reverseDiv(this.scalar());
        } else {
            return this.duplicate().divInPlace(that);
        }
    }

    @Override
    public TENSOR minusInPlace(T value) {
        return set(tensor.subi(value));
    }

    @Override
    public TENSOR plusInPlace(T value) {
        return set(tensor.addi(value));
    }

    @Override
    public TENSOR timesInPlace(T value) {
        return set(tensor.muli(value));
    }

    @Override
    public TENSOR divInPlace(T value) {
        return set(tensor.divi(value));
    }

    @Override
    public TENSOR powInPlace(TENSOR exponent) {
        if (exponent.isScalar()) {
            Transforms.pow(tensor, exponent.scalar(), false);
        } else {
            INDArray exponentArray = getTensor(exponent);
            tensor = INDArrayShim.pow(tensor, exponentArray);
        }
        return set(tensor);
    }

    @Override
    public TENSOR powInPlace(T exponent) {
        Transforms.pow(tensor, exponent, false);
        return set(tensor);
    }

    /**
     * @param that Right operand.
     * @return A new DoubleTensor instance only if <i>this</i> has a length of 1 and right operand has a length greater than 1.
     * Otherwise return <i>this</i>.
     */
    @Override
    public TENSOR minusInPlace(TENSOR that) {
        if (that.isScalar()) {
            tensor.subi(that.scalar());
        } else if (this.isScalar()) {
            return this.minus(that);
        } else {
            INDArray result = INDArrayShim.subi(tensor, getTensor(that).dup());
            if (result != tensor) {
                return create(result);
            }
        }
        return set(tensor);
    }

    @Override
    public TENSOR reverseMinusInPlace(TENSOR that) {
        if (that.isScalar()) {
            tensor.subi(that.scalar());
        } else if (this.isScalar()) {
            return this.minus(that);
        } else {
            INDArray result = INDArrayShim.rsubi(tensor, getTensor(that).dup());
            if (result != tensor) {
                return create(result);
            }
        }
        return set(tensor);
    }

    @Override
    public TENSOR reverseMinusInPlace(T value) {
        return set(tensor.rsubi(value));
    }

    /**
     * @param that Right operand.
     * @return A new DoubleTensor instance only if <i>this</i> has a length of 1 and right operand has a length greater than 1.
     * Otherwise return <i>this</i>.
     */
    @Override
    public TENSOR plusInPlace(TENSOR that) {
        if (that.isScalar()) {
            tensor.addi(that.scalar());
        } else if (this.isScalar()) {
            return this.plus(that);
        } else {
            INDArray result = INDArrayShim.addi(tensor, getTensor(that).dup());
            if (result != tensor) {
                return create(result);
            }
        }
        return set(tensor);
    }

    /**
     * @param that Right operand.
     * @return A new DoubleTensor instance only if <i>this</i> has a length of 1 and right operand has a length greater than 1.
     * Otherwise return <i>this</i>.
     */
    @Override
    public TENSOR timesInPlace(TENSOR that) {
        if (that.isScalar()) {
            tensor.muli(that.scalar());
        } else if (this.isScalar()) {
            return this.times(that);
        } else {
            INDArray result = INDArrayShim.muli(tensor, getTensor(that).dup());
            if (result != tensor) {
                return create(result);
            }
        }
        return set(tensor);
    }

    /**
     * @param that Right operand.
     * @return A new DoubleTensor instance only if <i>this</i> has a length of 1 and right operand has a length greater than 1.
     * Otherwise return <i>this</i>.
     */
    @Override
    public TENSOR divInPlace(TENSOR that) {
        if (that.isScalar()) {
            tensor.divi(that.scalar());
        } else if (this.isScalar()) {
            return this.div(that);
        } else {
            INDArray result = INDArrayShim.divi(tensor, getTensor(that).dup());
            if (result != tensor) {
                return create(result);
            }
        }
        return set(tensor);
    }

    @Override
    public TENSOR reverseDivInPlace(T value) {
        return set(tensor.rdivi(value));
    }

    @Override
    public TENSOR reverseDivInPlace(TENSOR that) {
        if (that.isScalar()) {
            tensor.subi(that.scalar());
        } else if (this.isScalar()) {
            return this.minus(that);
        } else {
            INDArray result = INDArrayShim.rdivi(tensor, getTensor(that).dup());
            if (result != tensor) {
                return create(result);
            }
        }
        return set(tensor);
    }

    @Override
    public TENSOR unaryMinusInPlace() {
        return set(tensor.negi());
    }

    @Override
    public TENSOR absInPlace() {
        Transforms.abs(tensor, false);
        return set(tensor);
    }

    @Override
    public TENSOR signInPlace() {
        Transforms.sign(tensor, false);
        return set(tensor);
    }

    @Override
    public BooleanTensor greaterThan(T value) {
        return fromMask(tensor.gt(value));
    }

    @Override
    public BooleanTensor greaterThanOrEqual(T value) {
        return fromMask(tensor.gte(value));
    }

    @Override
    public BooleanTensor greaterThan(TENSOR value) {

        INDArray mask;
        if (value.isScalar()) {
            mask = tensor.gt(value.scalar());
        } else {
            INDArray indArray = getTensor(value);
            mask = INDArrayShim.gt(tensor, indArray);
        }

        return fromMask(mask);
    }

    @Override
    public BooleanTensor greaterThanOrEqual(TENSOR value) {

        INDArray mask;
        if (value.isScalar()) {
            mask = tensor.gte(value.scalar());
        } else {
            INDArray indArray = getTensor(value);
            mask = tensor.dup();
            mask = INDArrayShim.gte(mask, indArray);
        }

        return fromMask(mask);
    }

    @Override
    public BooleanTensor lessThan(T value) {
        return fromMask(tensor.lt(value));
    }

    @Override
    public BooleanTensor lessThanOrEqual(T value) {
        return fromMask(tensor.lte(value));
    }

    @Override
    public BooleanTensor lessThan(TENSOR value) {

        INDArray mask;
        if (value.isScalar()) {
            mask = tensor.lt(value.scalar());
        } else {
            INDArray indArray = getTensor(value);
            mask = INDArrayShim.lt(tensor, indArray);
        }

        return fromMask(mask);
    }

    @Override
    public BooleanTensor lessThanOrEqual(TENSOR value) {

        INDArray mask;
        if (value.isScalar()) {
            mask = tensor.lte(value.scalar());
        } else {

            INDArray indArray = getTensor(value);
            mask = tensor.dup();
            mask = INDArrayShim.lte(mask, indArray);
        }

        return fromMask(mask);
    }

    @Override
    public TENSOR applyInPlace(Function<T, T> function) {
        FlattenedView<T> flattenedView = getFlattenedView();
        for (int i = 0; i < flattenedView.size(); i++) {
            flattenedView.set(i, function.apply(flattenedView.get(i)));
        }
        return getThis();
    }

    @Override
    public BooleanTensor elementwiseEquals(TENSOR that) {
        if (isScalar()) {
            return that.elementwiseEquals(this.scalar());
        } else if (that.isScalar()) {
            return elementwiseEquals(that.scalar());
        } else {
            INDArray mask = INDArrayShim.eq(tensor, getTensor(that));
            return fromMask(mask);
        }
    }

    @Override
    public BooleanTensor elementwiseEquals(T value) {
        return fromMask(tensor.eq(value));
    }

    @Override
    public BooleanTensor equalsWithinEpsilon(TENSOR o, T epsilon) {
        return this.minus(o).absInPlace().lessThanOrEqual(epsilon);
    }

    @Override
    public TENSOR tensorMultiply(TENSOR value, int[] dimLeft, int[] dimsRight) {
        return TensorMulByMatrixMul.tensorMmul((TENSOR) this, value, dimLeft, dimsRight);
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
    public long[] asFlatLongArray() {
        return tensor.dup().data().asLong();
    }

    @Override
    public BooleanTensor toBoolean() {
        return BooleanTensor.create(asBoolean(tensor.castTo(DataType.BOOL)), getShape());
    }

    @Override
    public DoubleTensor toDouble() {
        return new Nd4jDoubleTensor(tensor.castTo(DataType.DOUBLE));
    }

    @Override
    public IntegerTensor toInteger() {
        return new Nd4jIntegerTensor(tensor.castTo(DataType.INT));
    }

    @Override
    public LongTensor toLong() {
        return LongTensor.create(tensor.dup().data().asLong(), tensor.shape());
    }

    protected final BooleanTensor fromMask(INDArray mask) {
        long[] shape = mask.shape();
        DataBuffer data = mask.data();
        boolean[] boolsFromMask = new boolean[checkedCast(mask.length())];

        for (int i = 0; i < boolsFromMask.length; i++) {
            boolsFromMask[i] = data.getNumber(i).intValue() != 0;
        }
        return JVMBooleanTensor.create(boolsFromMask, shape);
    }

    protected abstract T getNumber(Number number);

}
