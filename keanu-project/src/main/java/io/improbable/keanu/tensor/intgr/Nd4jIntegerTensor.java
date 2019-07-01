package io.improbable.keanu.tensor.intgr;

import com.google.common.base.Preconditions;
import io.improbable.keanu.tensor.INDArrayExtensions;
import io.improbable.keanu.tensor.INDArrayShim;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.TypedINDArrayFactory;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.bool.JVMBooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.comparison.CompareAndSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

import static com.google.common.primitives.Ints.checkedCast;

/**
 * Class for representing n-dimensional arrays of integers. This is
 * backed by Nd4j which stores the int as a double.
 */
public class Nd4jIntegerTensor implements IntegerTensor {

    static {
        INDArrayShim.startNewThreadForNd4j();
    }

    private static final DataType BUFFER_TYPE = DataType.INT;
    private INDArray tensor;

    public Nd4jIntegerTensor(int[] data, long[] shape) {
        this(TypedINDArrayFactory.create(data, shape));
    }

    public Nd4jIntegerTensor(INDArray tensor) {
        this.tensor = tensor;
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

    static INDArray unsafeGetNd4J(IntegerTensor that) {
        if (that.isLengthOne()) {
            return Nd4j.scalar(that.scalar()).reshape(that.getShape());
        }
        return ((Nd4jIntegerTensor) that).tensor;
    }

    @Override
    public IntegerTensor reshape(long... newShape) {
        return new Nd4jIntegerTensor(tensor.reshape(newShape));
    }

    @Override
    public IntegerTensor permute(int... rearrange) {
        return new Nd4jIntegerTensor(tensor.permute(rearrange));
    }

    @Override
    public IntegerTensor broadcast(long... toShape) {
        return new Nd4jIntegerTensor(tensor.broadcast(toShape));
    }

    @Override
    public IntegerTensor diag() {
        return new Nd4jIntegerTensor(Nd4j.diag(tensor));
    }

    @Override
    public IntegerTensor transpose() {
        Preconditions.checkArgument(isMatrix(), "Cannot transpose rank " + getRank());
        return new Nd4jIntegerTensor(tensor.transpose());
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
        return null;
    }

    @Override
    public boolean equalsWithinEpsilon(IntegerTensor other, Integer epsilon) {
        return false;
    }

    @Override
    public IntegerTensor matrixMultiply(IntegerTensor value) {
        INDArray mmulResult = tensor.mmul(unsafeGetNd4J(value));
        return new Nd4jIntegerTensor(mmulResult);
    }

    @Override
    public IntegerTensor tensorMultiply(IntegerTensor value, int[] dimLeft, int[] dimsRight) {
        INDArray tensorMmulResult = Nd4j.tensorMmul(tensor, unsafeGetNd4J(value), new int[][]{dimLeft, dimsRight});
        return new Nd4jIntegerTensor(tensorMmulResult);
    }

    @Override
    public IntegerTensor setWithMaskInPlace(IntegerTensor mask, Integer value) {
        if (this.getLength() != mask.getLength()) {
            throw new IllegalArgumentException("The lengths of the tensor and mask must match, but got tensor length: " + this.getLength() + ", mask length: " + mask.getLength());
        }
        INDArray maskDup = unsafeGetNd4J(mask);

        INDArray dblBuffer = tensor.castTo(DataType.DOUBLE);
        INDArray dblMask = maskDup.castTo(DataType.DOUBLE);

        double trueValue = 1.0;
        if (value == 0.0) {
            trueValue = 1.0 - trueValue;
            dblMask.negi().addi(1.0);
        }
        double falseValue = 1.0 - trueValue;

        Nd4j.getExecutioner().exec(
            new CompareAndSet(dblMask, value, Conditions.equals(trueValue))
        );

        Nd4j.getExecutioner().exec(
            new CompareAndSet(dblBuffer, dblMask, Conditions.notEquals(falseValue))
        );

        tensor = dblBuffer.castTo(DataType.INT);

        return this;
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
            INDArray exponentArray = unsafeGetNd4J(exponent);
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
            INDArray result = INDArrayShim.subi(tensor, unsafeGetNd4J(that).dup());
            if (result != tensor) {
                return new Nd4jIntegerTensor(result);
            }
        }
        return this;
    }

    @Override
    public IntegerTensor reverseMinusInPlace(IntegerTensor value) {
        tensor.rsubi(unsafeGetNd4J(value));
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
            INDArray result = INDArrayShim.addi(tensor, unsafeGetNd4J(that).dup());
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
            INDArray result = INDArrayShim.muli(tensor, unsafeGetNd4J(that).dup());
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
            INDArray result = INDArrayShim.divi(tensor, unsafeGetNd4J(that).dup());
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
        tensor.rdivi(unsafeGetNd4J(value));
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
        return null;
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
            INDArray indArray = unsafeGetNd4J(value);
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
            INDArray indArray = unsafeGetNd4J(value);
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
            tensor = INDArrayShim.min(tensor, unsafeGetNd4J(min));
        }
        return this;
    }

    @Override
    public IntegerTensor maxInPlace(IntegerTensor max) {
        if (max.isScalar()) {
            Transforms.max(tensor, max.scalar(), false);
        } else {
            tensor = INDArrayShim.max(tensor, unsafeGetNd4J(max));
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
            INDArray indArray = unsafeGetNd4J(value);
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
            INDArray indArray = unsafeGetNd4J(value);
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
    public int getRank() {
        return tensor.rank();
    }

    @Override
    public long[] getShape() {
        return tensor.shape();
    }

    @Override
    public long[] getStride() {
        return tensor.stride();
    }

    @Override
    public long getLength() {
        return tensor.length();
    }

    @Override
    public IntegerTensor duplicate() {
        return new Nd4jIntegerTensor(tensor.dup());
    }

    @Override
    public BooleanTensor elementwiseEquals(Tensor that) {
        if (that instanceof IntegerTensor) {
            if (this.isScalar()) {
                return ((IntegerTensor) that).elementwiseEquals(this.scalar());
            } else if (that.isScalar()) {
                return elementwiseEquals(((IntegerTensor) that).scalar());
            } else {
                INDArray mask = INDArrayShim.eq(tensor, unsafeGetNd4J((IntegerTensor) that));
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
    public boolean equals(Object o) {
        if (this == o) return true;

        if (o instanceof Nd4jIntegerTensor) {
            return tensor.equals(((Nd4jIntegerTensor) o).tensor);
        } else if (o instanceof Tensor) {
            Tensor that = (Tensor) o;
            if (!Arrays.equals(that.getShape(), getShape())) return false;
            return Arrays.equals(
                that.asFlatArray(),
                this.asFlatArray()
            );
        }

        return false;
    }

    @Override
    public int hashCode() {
        int result = tensor != null ? tensor.hashCode() : 0;
        result = 31 * result + Arrays.hashCode(getShape());
        return result;
    }

    @Override
    public String toString() {
        return tensor.toString();
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
        tensor.fmodi(unsafeGetNd4J(that));
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
