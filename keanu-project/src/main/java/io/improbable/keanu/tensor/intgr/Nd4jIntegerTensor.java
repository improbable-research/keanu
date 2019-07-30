package io.improbable.keanu.tensor.intgr;

import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.tensor.ndj4.INDArrayShim;
import io.improbable.keanu.tensor.ndj4.Nd4jFixedPointTensor;
import io.improbable.keanu.tensor.ndj4.Nd4jTensor;
import io.improbable.keanu.tensor.ndj4.TypedINDArrayFactory;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.NotImplementedException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static io.improbable.keanu.tensor.TensorShape.getBroadcastResultShape;
import static java.util.Arrays.copyOf;

/**
 * Class for representing n-dimensional arrays of integers. This is
 * backed by Nd4j.
 */
public class Nd4jIntegerTensor extends Nd4jFixedPointTensor<Integer, IntegerTensor> implements IntegerTensor {

    static {
        INDArrayShim.startNewThreadForNd4j();
    }

    private static final DataType BUFFER_TYPE = DataType.INT;

    public Nd4jIntegerTensor(int[] data, long[] shape) {
        this(TypedINDArrayFactory.create(data, shape));
    }

    public Nd4jIntegerTensor(INDArray tensor) {
        super(tensor);
    }

    @Override
    protected INDArray getTensor(Tensor<Integer, ?> tensor) {
        return getAsINDArray(tensor);
    }

    public static Nd4jIntegerTensor scalar(int scalarValue) {
        return new Nd4jIntegerTensor(Nd4j.scalar(scalarValue));
    }

    public static Nd4jIntegerTensor create(int[] values, long[] shape) {
        return new Nd4jIntegerTensor(values, shape);
    }

    public static Nd4jIntegerTensor create(int value, long[] shape) {
        return new Nd4jIntegerTensor(Nd4j.valueArrayOf(shape, value));
    }

    public static Nd4jIntegerTensor ones(long[] shape) {
        return new Nd4jIntegerTensor(TypedINDArrayFactory.ones(shape, BUFFER_TYPE));
    }

    public static Nd4jIntegerTensor eye(long n) {
        return new Nd4jIntegerTensor(TypedINDArrayFactory.eye(n, BUFFER_TYPE));
    }

    public static Nd4jIntegerTensor zeros(long[] shape) {
        return new Nd4jIntegerTensor(TypedINDArrayFactory.zeros(shape, BUFFER_TYPE));
    }

    public static Nd4jIntegerTensor arange(int start, int end) {
        return new Nd4jIntegerTensor(TypedINDArrayFactory.arange(start, end));
    }

    public static Nd4jDoubleTensor arange(int start, int end, int stepSize) {
        int stepCount = (end - start) / stepSize;
        INDArray arangeWithStep = TypedINDArrayFactory.arange(0, stepCount).muli(stepSize).addi(start);
        return new Nd4jDoubleTensor(arangeWithStep);
    }

    static INDArray getAsINDArray(Tensor that) {

        if (that instanceof Nd4jTensor) {
            INDArray array = ((Nd4jTensor) that).getTensor();
            if (array.dataType() == DataType.INT) {
                return array;
            } else {
                return array.castTo(DataType.INT);
            }
        } else if (that instanceof NumberTensor) {
            return TypedINDArrayFactory.create(((NumberTensor) that).toInteger().asFlatIntegerArray(), that.getShape());
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
    public IntegerTensor diag() {
        if (getRank() > 1) {
            return toDouble().diag().toInteger();
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
    public IntegerTensor diagPart() {
        if (tensor.size(0) != tensor.size(1)) {
            return toDouble().diagPart().toInteger();
        } else {
            return super.diagPart();
        }
    }

    @Override
    protected Integer getNumber(Number number) {
        return number.intValue();
    }

    @Override
    public IntegerTensor setAllInPlace(Integer value) {
        tensor = Nd4j.valueArrayOf(getShape(), value);
        return this;
    }

    @Override
    public IntegerTensor safeLogTimesInPlace(IntegerTensor y) {
        throw new NotImplementedException("");
    }

    @Override
    public IntegerTensor greaterThanMask(IntegerTensor greaterThanThis) {
        return greaterThan(greaterThanThis).toIntegerMask();
    }

    @Override
    public IntegerTensor greaterThanOrEqualToMask(IntegerTensor greaterThanOrEqualToThis) {
        return greaterThanOrEqual(greaterThanOrEqualToThis).toIntegerMask();
    }

    @Override
    public IntegerTensor lessThanMask(IntegerTensor lessThanThis) {
        return lessThan(lessThanThis).toIntegerMask();
    }

    @Override
    public IntegerTensor lessThanOrEqualToMask(IntegerTensor lessThanOrEqualToThis) {
        return lessThanOrEqual(lessThanOrEqualToThis).toIntegerMask();
    }

    @Override
    public DoubleTensor toDouble() {
        return new Nd4jDoubleTensor(tensor.castTo(DataType.DOUBLE));
    }

    @Override
    public IntegerTensor toInteger() {
        return duplicate();
    }

    @Override
    protected IntegerTensor create(INDArray tensor) {
        return new Nd4jIntegerTensor(tensor);
    }

    @Override
    protected IntegerTensor set(INDArray tensor) {
        this.tensor = tensor.dataType() == DataType.INT ? tensor : tensor.castTo(DataType.INT);
        return this;
    }

    @Override
    protected IntegerTensor getThis() {
        return this;
    }

    @Override
    public IntegerTensor where(BooleanTensor predicate, IntegerTensor els) {
        final long[] resultShape = getBroadcastResultShape(getBroadcastResultShape(getShape(), predicate.getShape()), els.getShape());
        final IntegerTensor broadcastedTrue = this.hasShape(resultShape) ? this : this.broadcast(resultShape);
        final IntegerTensor broadcastedFalse = els.hasShape(resultShape) ? els : els.broadcast(resultShape);
        final BooleanTensor broadcastedPredicate = predicate.hasShape(resultShape) ? predicate : predicate.broadcast(resultShape);

        FlattenedView<Integer> trueValuesFlattened = broadcastedTrue.getFlattenedView();
        FlattenedView<Integer> falseValuesFlattened = broadcastedFalse.getFlattenedView();
        FlattenedView<Boolean> predicateValuesFlattened = broadcastedPredicate.getFlattenedView();

        int[] result = new int[TensorShape.getLengthAsInt(resultShape)];
        for (int i = 0; i < result.length; i++) {
            result[i] = predicateValuesFlattened.get(i) ? trueValuesFlattened.get(i) : falseValuesFlattened.get(i);
        }

        return IntegerTensor.create(result, copyOf(resultShape, resultShape.length));
    }

    @Override
    public Integer[] asFlatArray() {
        return ArrayUtils.toObject(asFlatIntegerArray());
    }

    @Override
    public FlattenedView<Integer> getFlattenedView() {
        return new Nd4jIntegerFlattenedView();
    }

    private class Nd4jIntegerFlattenedView implements FlattenedView<Integer> {

        @Override
        public long size() {
            return tensor.data().length();
        }

        @Override
        public Integer get(long index) {
            return tensor.data().getInt(index);
        }

        @Override
        public Integer getOrScalar(long index) {
            if (tensor.length() == 1) {
                return get(0);
            } else {
                return get(index);
            }
        }

        @Override
        public void set(long index, Integer value) {
            tensor.data().put(index, value);
        }

    }

}
