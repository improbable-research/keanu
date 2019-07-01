package io.improbable.keanu.tensor.intgr;

import io.improbable.keanu.tensor.INDArrayExtensions;
import io.improbable.keanu.tensor.INDArrayShim;
import io.improbable.keanu.tensor.Nd4jFixedPointTensor;
import io.improbable.keanu.tensor.Nd4jTensor;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.TypedINDArrayFactory;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.bool.JVMBooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.NotImplementedException;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

import static com.google.common.primitives.Ints.checkedCast;

/**
 * Class for representing n-dimensional arrays of integers. This is
 * backed by Nd4j which stores the int as a double.
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
    protected INDArray getTensor(IntegerTensor tensor) {
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

    static INDArray getAsINDArray(IntegerTensor that) {

        if (that instanceof Nd4jTensor) {
            INDArray array = ((Nd4jTensor) that).getTensor();
            if (array.dataType() == DataType.INT) {
                return array;
            } else {
                return array.castTo(DataType.INT);
            }
        } else {
            return TypedINDArrayFactory.create(that.asFlatIntegerArray(), that.getShape());
        }
    }

    @Override
    public IntegerTensor sum(int... overDimensions) {
        return new Nd4jIntegerTensor(tensor.sum(overDimensions));
    }

    @Override
    public IntegerTensor cumSumInPlace(int dimension) {
        tensor.cumsumi(dimension);
        return this;
    }

    @Override
    public Integer product() {
        return tensor.prodNumber().intValue();
    }

    @Override
    public IntegerTensor clampInPlace(IntegerTensor min, IntegerTensor max) {
        return minInPlace(max).maxInPlace(min);
    }

    @Override
    public boolean equalsWithinEpsilon(IntegerTensor o, Integer epsilon) {
        if (this == o) return true;

        if (o instanceof Nd4jTensor) {
            return tensor.equalsWithEps(((Nd4jTensor) o).getTensor(), epsilon);
        } else {
            if (this.hasSameShapeAs(o)) {
                IntegerTensor difference = o.minus(this);
                return difference.abs().lessThan(epsilon).allTrue();
            }
        }
        return false;
    }

    @Override
    public IntegerTensor matrixMultiply(IntegerTensor value) {
        INDArray mmulResult = tensor.mmul(getAsINDArray(value));
        return new Nd4jIntegerTensor(mmulResult);
    }

    @Override
    public IntegerTensor tensorMultiply(IntegerTensor value, int[] dimLeft, int[] dimsRight) {
        INDArray tensorMmulResult = Nd4j.tensorMmul(tensor, getAsINDArray(value), new int[][]{dimLeft, dimsRight});
        return new Nd4jIntegerTensor(tensorMmulResult);
    }

    @Override
    public IntegerTensor minusInPlace(Integer value) {
        tensor.subi(value);
        return this;
    }

    @Override
    public IntegerTensor plusInPlace(Integer value) {
        tensor.addi(value);
        return this;
    }

    @Override
    public IntegerTensor timesInPlace(Integer value) {
        tensor.muli(value);
        return this;
    }

    @Override
    public IntegerTensor divInPlace(Integer value) {
        tensor.divi(value);
        INDArrayExtensions.castToInteger(tensor, false);
        return this;
    }

    @Override
    public IntegerTensor powInPlace(IntegerTensor exponent) {
        if (exponent.isScalar()) {
            Transforms.pow(tensor, exponent.scalar(), false);
        } else {
            INDArray exponentArray = getAsINDArray(exponent);
            tensor = INDArrayShim.pow(tensor, exponentArray);
        }
        return this;
    }

    @Override
    public IntegerTensor powInPlace(Integer exponent) {
        Transforms.pow(tensor, exponent, false);
        return this;
    }

    @Override
    public Integer average() {
        return (int) (tensor.sumNumber().doubleValue() / tensor.length());
    }

    @Override
    public Integer standardDeviation() {
        return tensor.stdNumber().intValue();
    }

    @Override
    public IntegerTensor minusInPlace(IntegerTensor that) {
        if (that.isScalar()) {
            tensor.subi(that.scalar());
        } else {
            INDArray result = INDArrayShim.subi(tensor, getAsINDArray(that).dup());
            if (result != tensor) {
                return new Nd4jIntegerTensor(result);
            }
        }
        return this;
    }

    @Override
    public IntegerTensor reverseMinusInPlace(IntegerTensor value) {
        tensor.rsubi(getAsINDArray(value));
        return this;
    }

    @Override
    public IntegerTensor reverseMinusInPlace(Integer value) {
        tensor.rsubi(value);
        return this;
    }

    @Override
    public IntegerTensor plusInPlace(IntegerTensor that) {
        if (that.isScalar()) {
            tensor.addi(that.scalar());
        } else {
            INDArray result = INDArrayShim.addi(tensor, getAsINDArray(that).dup());
            if (result != tensor) {
                return new Nd4jIntegerTensor(result);
            }
        }
        return this;
    }

    @Override
    public IntegerTensor timesInPlace(IntegerTensor that) {
        if (that.isScalar()) {
            tensor.muli(that.scalar());
        } else {
            INDArray result = INDArrayShim.muli(tensor, getAsINDArray(that).dup());
            if (result != tensor) {
                return new Nd4jIntegerTensor(result);
            }
        }
        return this;
    }

    @Override
    public IntegerTensor divInPlace(IntegerTensor that) {
        if (that.isScalar()) {
            tensor.divi(that.scalar());
        } else {
            INDArray result = INDArrayShim.divi(tensor, getAsINDArray(that).dup());
            if (result != tensor) {
                return new Nd4jIntegerTensor(INDArrayExtensions.castToInteger(result, false));
            }
        }
        INDArrayExtensions.castToInteger(tensor, false);
        return this;
    }

    @Override
    public IntegerTensor reverseDivInPlace(Integer value) {
        tensor.rdivi(value);
        return this;
    }

    @Override
    public IntegerTensor reverseDivInPlace(IntegerTensor value) {
        tensor.rdivi(getAsINDArray(value));
        return this;
    }

    @Override
    public IntegerTensor unaryMinusInPlace() {
        tensor.negi();
        return this;
    }

    @Override
    public IntegerTensor absInPlace() {
        Transforms.abs(tensor, false);
        return this;
    }

    @Override
    public IntegerTensor applyInPlace(Function<Integer, Integer> function) {
        DataBuffer data = tensor.data();
        for (int i = 0; i < data.length(); i++) {
            data.put(i, function.apply(data.getInt(i)));
        }
        return this;
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
    public BooleanTensor lessThan(Integer value) {
        return fromMask(tensor.lt(value));
    }

    @Override
    public IntegerTensor getGreaterThanMask(IntegerTensor greaterThanThis) {
        return greaterThan(greaterThanThis).toIntegerMask();
    }

    @Override
    public IntegerTensor getGreaterThanOrEqualToMask(IntegerTensor greaterThanOrEqualToThis) {
        return greaterThanOrEqual(greaterThanOrEqualToThis).toIntegerMask();
    }

    @Override
    public IntegerTensor getLessThanMask(IntegerTensor lessThanThis) {
        return lessThan(lessThanThis).toIntegerMask();
    }

    @Override
    public IntegerTensor getLessThanOrEqualToMask(IntegerTensor lessThanOrEqualToThis) {
        return lessThanOrEqual(lessThanOrEqualToThis).toIntegerMask();
    }

    @Override
    public BooleanTensor lessThanOrEqual(Integer value) {
        return fromMask(tensor.lte(value));
    }

    @Override
    public BooleanTensor lessThan(IntegerTensor value) {

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
    public BooleanTensor lessThanOrEqual(IntegerTensor value) {

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
    public BooleanTensor greaterThan(Integer value) {
        return fromMask(tensor.gt(value));
    }

    @Override
    public BooleanTensor greaterThanOrEqual(Integer value) {
        return fromMask(tensor.gte(value));
    }

    @Override
    public IntegerTensor minInPlace(IntegerTensor min) {
        if (min.isScalar()) {
            Transforms.min(tensor, min.scalar(), false);
        } else {
            tensor = INDArrayShim.min(tensor, getAsINDArray(min));
        }
        return this;
    }

    @Override
    public IntegerTensor maxInPlace(IntegerTensor max) {
        if (max.isScalar()) {
            Transforms.max(tensor, max.scalar(), false);
        } else {
            tensor = INDArrayShim.max(tensor, getAsINDArray(max));
        }
        return this;
    }

    @Override
    public Integer min() {
        return tensor.minNumber().intValue();
    }

    @Override
    public Integer max() {
        return tensor.maxNumber().intValue();
    }

    @Override
    public int argMax() {
        return tensor.argMax().getInt(0);
    }

    @Override
    public IntegerTensor argMax(int axis) {
        long[] shape = this.getShape();
        TensorShapeValidation.checkDimensionExistsInShape(axis, shape);
        return new Nd4jIntegerTensor(tensor.argMax(axis).reshape(TensorShape.removeDimension(axis, shape)));
    }

    @Override
    public int argMin() {
        return Nd4j.argMin(tensor).getInt(0);
    }

    @Override
    public IntegerTensor argMin(int axis) {
        long[] shape = this.getShape();
        TensorShapeValidation.checkDimensionExistsInShape(axis, shape);
        return new Nd4jIntegerTensor(Nd4j.argMin(tensor, axis).reshape(TensorShape.removeDimension(axis, shape)));
    }

    @Override
    public BooleanTensor greaterThan(IntegerTensor value) {

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
    public BooleanTensor greaterThanOrEqual(IntegerTensor value) {

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
    public Integer sum() {
        return tensor.sumNumber().intValue();
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
    public BooleanTensor elementwiseEquals(Tensor that) {
        if (that instanceof IntegerTensor) {
            if (this.isScalar()) {
                return ((IntegerTensor) that).elementwiseEquals(this.scalar());
            } else if (that.isScalar()) {
                return elementwiseEquals(((IntegerTensor) that).scalar());
            } else {
                INDArray mask = INDArrayShim.eq(tensor, getAsINDArray((IntegerTensor) that));
                return fromMask(mask);
            }
        } else {
            return Tensor.elementwiseEquals(this, that);
        }
    }

    @Override
    public BooleanTensor elementwiseEquals(Integer value) {
        return fromMask(tensor.eq(value));
    }

    @Override
    public IntegerTensor slice(int dimension, long index) {
        return new Nd4jIntegerTensor(tensor.slice(index, dimension));
    }

    @Override
    public IntegerTensor take(long... index) {
        return scalar(getValue(index));
    }

    @Override
    public List<IntegerTensor> split(int dimension, long... splitAtIndices) {

        List<INDArray> splitINDArrays = INDArrayExtensions.split(tensor, dimension, splitAtIndices);

        return splitINDArrays.stream()
            .map(Nd4jIntegerTensor::new)
            .collect(Collectors.toList());
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

    private BooleanTensor fromMask(INDArray mask) {
        long[] shape = mask.shape();
        DataBuffer data = mask.data();
        boolean[] boolsFromMask = new boolean[checkedCast(mask.length())];

        for (int i = 0; i < boolsFromMask.length; i++) {
            boolsFromMask[i] = data.getInt(i) != 0;
        }
        return JVMBooleanTensor.create(boolsFromMask, shape);
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
    public Integer[] asFlatArray() {
        return ArrayUtils.toObject(asFlatIntegerArray());
    }

    @Override
    public IntegerTensor modInPlace(Integer that) {
        tensor.fmodi(that);
        return this;
    }

    @Override
    public IntegerTensor modInPlace(IntegerTensor that) {
        tensor.fmodi(getAsINDArray(that));
        return this;
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
