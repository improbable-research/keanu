package io.improbable.keanu.tensor.bool;

import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import io.improbable.keanu.tensor.JVMTensorBroadcast;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.buffer.BooleanBuffer;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import org.apache.commons.lang3.ArrayUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.function.BiFunction;

import static io.improbable.keanu.tensor.JVMTensorBroadcast.broadcastIfNeeded;
import static io.improbable.keanu.tensor.TensorShape.convertFromFlatIndexToPermutedFlatIndex;
import static io.improbable.keanu.tensor.TensorShape.getAbsoluteDimension;
import static io.improbable.keanu.tensor.TensorShape.getFlatIndex;
import static io.improbable.keanu.tensor.TensorShape.getPermutationForDimensionToDimensionZero;
import static io.improbable.keanu.tensor.TensorShape.getPermutedIndices;
import static io.improbable.keanu.tensor.TensorShape.getReshapeAllowingWildcard;
import static io.improbable.keanu.tensor.TensorShape.getRowFirstStride;
import static io.improbable.keanu.tensor.TensorShape.getShapeIndices;
import static io.improbable.keanu.tensor.TensorShape.invertedPermute;
import static io.improbable.keanu.tensor.bool.BroadcastableBooleanOperations.AND;
import static io.improbable.keanu.tensor.bool.BroadcastableBooleanOperations.OR;
import static io.improbable.keanu.tensor.bool.BroadcastableBooleanOperations.XOR;
import static java.util.Arrays.copyOf;

public class JVMBooleanTensor implements BooleanTensor {

    private static final BooleanBuffer.BooleanArrayWrapperFactory factory = new BooleanBuffer.BooleanArrayWrapperFactory();

    private BooleanBuffer.PrimitiveBooleanWrapper buffer;
    private long[] shape;
    private long[] stride;

    private JVMBooleanTensor(BooleanBuffer.PrimitiveBooleanWrapper buffer, long[] shape, long[] stride) {
        this.buffer = buffer;
        this.shape = shape;
        this.stride = stride;
    }

    private JVMBooleanTensor(BooleanBuffer.PrimitiveBooleanWrapper buffer, long[] shape) {
        this.buffer = buffer;
        this.shape = shape;
        this.stride = getRowFirstStride(shape);
    }

    /**
     * @param buffer tensor buffer used c ordering
     * @param shape  desired shape of tensor
     */
    private JVMBooleanTensor(boolean[] buffer, long[] shape) {
        this.buffer = factory.create(buffer);
        this.shape = shape;
        this.stride = getRowFirstStride(shape);
    }

    private JVMBooleanTensor(boolean[] buffer, long[] shape, long[] stride) {
        this.buffer = factory.create(buffer);
        this.shape = shape;
        this.stride = stride;
    }

    /**
     * @param constant constant boolean value to fill shape
     */
    private JVMBooleanTensor(boolean constant) {
        this.buffer = new BooleanBuffer.BooleanWrapper(constant);
        this.shape = Tensor.SCALAR_SHAPE;
        this.stride = Tensor.SCALAR_STRIDE;
    }

    public static JVMBooleanTensor scalar(boolean scalarValue) {
        return new JVMBooleanTensor(scalarValue);
    }

    public static JVMBooleanTensor create(boolean[] values, long... shape) {
        Preconditions.checkArgument(
            TensorShape.getLength(shape) == values.length,
            "Shape " + Arrays.toString(shape) + " does not match data length " + values.length
        );
        return new JVMBooleanTensor(values, shape);
    }

    public static JVMBooleanTensor create(boolean value, long... shape) {
        boolean[] buffer = new boolean[TensorShape.getLengthAsInt(shape)];

        if (value) {
            Arrays.fill(buffer, value);
        }

        return new JVMBooleanTensor(buffer, shape);
    }

    private BooleanBuffer.PrimitiveBooleanWrapper bufferCopy() {
        return buffer.copy();
    }

    @Override
    public BooleanTensor reshape(long... newShape) {
        return new JVMBooleanTensor(bufferCopy(), getReshapeAllowingWildcard(shape, buffer.getLength(), newShape));
    }

    @Override
    public BooleanTensor permute(int... rearrange) {
        Preconditions.checkArgument(rearrange.length == shape.length);
        long[] resultShape = getPermutedIndices(shape, rearrange);
        long[] resultStride = getRowFirstStride(resultShape);
        boolean[] newBuffer = new boolean[buffer.getLength()];

        for (int flatIndex = 0; flatIndex < buffer.getLength(); flatIndex++) {

            int permutedFlatIndex = convertFromFlatIndexToPermutedFlatIndex(
                flatIndex,
                shape, stride,
                resultShape, resultStride,
                rearrange
            );

            newBuffer[permutedFlatIndex] = buffer.get(flatIndex);
        }

        return new JVMBooleanTensor(newBuffer, resultShape);
    }

    @Override
    public BooleanTensor broadcast(long... toShape) {
        int outputLength = TensorShape.getLengthAsInt(toShape);
        long[] outputStride = TensorShape.getRowFirstStride(toShape);
        BooleanBuffer.PrimitiveBooleanWrapper outputBuffer = factory.createNew(outputLength);

        JVMTensorBroadcast.broadcast(buffer, shape, stride, outputBuffer, outputStride);

        return new JVMBooleanTensor(outputBuffer, toShape, outputStride);
    }

    public static BooleanTensor concat(int dimension, BooleanTensor... toConcat) {

        long[] concatShape = TensorShape.getConcatResultShape(dimension, toConcat);

        boolean shouldRearrange = dimension != 0;

        if (shouldRearrange) {

            int[] rearrange = getPermutationForDimensionToDimensionZero(dimension, concatShape);

            BooleanTensor[] toConcatOnDimensionZero = new BooleanTensor[toConcat.length];

            for (int i = 0; i < toConcatOnDimensionZero.length; i++) {
                toConcatOnDimensionZero[i] = toConcat[i].permute(rearrange);
            }

            long[] permutedConcatShape = getPermutedIndices(concatShape, rearrange);
            BooleanTensor concatOnDimZero = concatOnDimensionZero(permutedConcatShape, toConcatOnDimensionZero);

            return concatOnDimZero.permute(invertedPermute(rearrange));
        } else {

            return concatOnDimensionZero(concatShape, toConcat);
        }
    }

    private static JVMBooleanTensor concatOnDimensionZero(long[] concatShape, BooleanTensor... toConcat) {

        boolean[] concatBuffer = new boolean[TensorShape.getLengthAsInt(concatShape)];
        int bufferPosition = 0;

        for (int i = 0; i < toConcat.length; i++) {

            BooleanBuffer.PrimitiveBooleanWrapper cBuffer = getRawBufferIfJVMTensor(toConcat[i]);
            System.arraycopy(cBuffer.asBooleanArray(), 0, concatBuffer, bufferPosition, cBuffer.getLength());
            bufferPosition += cBuffer.getLength();
        }

        return new JVMBooleanTensor(concatBuffer, concatShape);
    }

    private static BooleanBuffer.PrimitiveBooleanWrapper getRawBufferIfJVMTensor(BooleanTensor tensor) {
        if (tensor instanceof JVMBooleanTensor) {
            return ((JVMBooleanTensor) tensor).buffer;
        } else {
            return factory.create(tensor.asFlatBooleanArray());
        }
    }

    @Override
    public DoubleTensor doubleWhere(DoubleTensor trueValue, DoubleTensor falseValue) {
        double[] trueValues = trueValue.asFlatDoubleArray();
        double[] falseValues = falseValue.asFlatDoubleArray();

        double[] result = new double[buffer.getLength()];
        for (int i = 0; i < result.length; i++) {
            result[i] = buffer.get(i) ? getOrScalar(trueValues, i) : getOrScalar(falseValues, i);
        }

        return DoubleTensor.create(result, copyOf(shape, shape.length));
    }

    @Override
    public IntegerTensor integerWhere(IntegerTensor trueValue, IntegerTensor falseValue) {
        FlattenedView<Integer> trueValuesFlattened = trueValue.getFlattenedView();
        FlattenedView<Integer> falseValuesFlattened = falseValue.getFlattenedView();

        int[] result = new int[buffer.getLength()];
        for (int i = 0; i < result.length; i++) {
            result[i] = buffer.get(i) ? trueValuesFlattened.getOrScalar(i) : falseValuesFlattened.getOrScalar(i);
        }

        return IntegerTensor.create(result, copyOf(shape, shape.length));
    }

    @Override
    public BooleanTensor booleanWhere(BooleanTensor trueValue, BooleanTensor falseValue) {
        FlattenedView<Boolean> trueValuesFlattened = trueValue.getFlattenedView();
        FlattenedView<Boolean> falseValuesFlattened = falseValue.getFlattenedView();

        boolean[] result = new boolean[buffer.getLength()];
        for (int i = 0; i < result.length; i++) {
            result[i] = buffer.get(i) ? trueValuesFlattened.getOrScalar(i) : falseValuesFlattened.getOrScalar(i);
        }

        return BooleanTensor.create(result, copyOf(shape, shape.length));
    }

    @Override
    public <T, TENSOR extends Tensor<T, TENSOR>> TENSOR where(TENSOR trueValue, TENSOR falseValue) {
        if (trueValue instanceof DoubleTensor && falseValue instanceof DoubleTensor) {
            return (TENSOR) doubleWhere((DoubleTensor) trueValue, (DoubleTensor) falseValue);
        } else if (trueValue instanceof IntegerTensor && falseValue instanceof IntegerTensor) {
            return (TENSOR) integerWhere((IntegerTensor) trueValue, (IntegerTensor) falseValue);
        } else if (trueValue instanceof BooleanTensor && falseValue instanceof BooleanTensor) {
            return (TENSOR) booleanWhere((BooleanTensor) trueValue, (BooleanTensor) falseValue);
        } else {
            FlattenedView<T> trueValuesFlattened = trueValue.getFlattenedView();
            FlattenedView<T> falseValuesFlattened = falseValue.getFlattenedView();

            T[] result = (T[]) (new Object[buffer.getLength()]);
            for (int i = 0; i < result.length; i++) {
                result[i] = buffer.get(i) ? trueValuesFlattened.getOrScalar(i) : falseValuesFlattened.getOrScalar(i);
            }

            return Tensor.create(result, copyOf(shape, shape.length));
        }
    }

    private double getOrScalar(double[] values, int index) {
        if (values.length == 1) {
            return values[0];
        } else {
            return values[index];
        }
    }

    @Override
    public BooleanTensor andInPlace(BooleanTensor that) {
        return binaryBooleanOpWithAutoBroadcast(that, AND, true);
    }

    @Override
    public BooleanTensor andInPlace(boolean that) {
        if (!that) {
            buffer.applyRight((l, r) -> r, false);
        }
        return this;
    }

    @Override
    public BooleanTensor orInPlace(BooleanTensor that) {
        return binaryBooleanOpWithAutoBroadcast(that, OR, true);
    }

    @Override
    public BooleanTensor orInPlace(boolean that) {
        if (that) {
            buffer.applyRight((l, r) -> r, true);
        }
        return this;
    }

    @Override
    public BooleanTensor xorInPlace(BooleanTensor that) {
        return binaryBooleanOpWithAutoBroadcast(that, XOR, true);
    }

    @Override
    public BooleanTensor notInPlace() {
        buffer.apply(v -> !v);
        return this;
    }

    @Override
    public boolean allTrue() {
        for (int i = 0; i < buffer.getLength(); i++) {
            if (!buffer.get(i)) {
                return false;
            }
        }
        return true;
    }

    @Override
    public boolean allFalse() {
        for (int i = 0; i < buffer.getLength(); i++) {
            if (buffer.get(i)) {
                return false;
            }
        }
        return true;
    }

    @Override
    public boolean anyTrue() {
        return !allFalse();
    }

    @Override
    public boolean anyFalse() {
        return !allTrue();
    }

    private BooleanTensor binaryBooleanOpWithAutoBroadcast(BooleanTensor right,
                                                           BiFunction<Boolean, Boolean, Boolean> op,
                                                           boolean inPlace) {
        final BooleanBuffer.PrimitiveBooleanWrapper rightBuffer = getRawBufferIfJVMTensor(right);
        final long[] rightShape = right.getShape();

        final JVMTensorBroadcast.ResultWrapper<Boolean, BooleanBuffer.PrimitiveBooleanWrapper> result = broadcastIfNeeded(
            factory, buffer, shape, stride, buffer.getLength(),
            rightBuffer, rightShape, right.getStride(), rightBuffer.getLength(),
            op, inPlace
        );

        if (inPlace) {
            this.buffer = result.outputBuffer;
            this.shape = result.outputShape;
            this.stride = result.outputStride;

            return this;
        } else {
            return new JVMBooleanTensor(result.outputBuffer, result.outputShape, result.outputStride);
        }
    }

    @Override
    public DoubleTensor toDoubleMask() {
        double[] doubles = asFlatDoubleArray();
        return DoubleTensor.create(doubles, copyOf(shape, shape.length));
    }

    @Override
    public IntegerTensor toIntegerMask() {
        int[] doubles = asFlatIntegerArray();
        return IntegerTensor.create(doubles, copyOf(shape, shape.length));
    }

    @Override
    public BooleanTensor slice(int dimension, long index) {
        Preconditions.checkArgument(dimension < shape.length && index < shape[dimension]);
        long[] resultShape = ArrayUtils.remove(shape, dimension);
        long[] resultStride = getRowFirstStride(resultShape);
        boolean[] newBuffer = new boolean[TensorShape.getLengthAsInt(resultShape)];

        for (int i = 0; i < newBuffer.length; i++) {

            long[] shapeIndices = ArrayUtils.insert(dimension, getShapeIndices(resultShape, resultStride, i), index);

            int j = Ints.checkedCast(getFlatIndex(shape, stride, shapeIndices));

            newBuffer[i] = buffer.get(j);
        }

        return new JVMBooleanTensor(newBuffer, resultShape);
    }

    @Override
    public BooleanTensor take(long... index) {
        return new JVMBooleanTensor(getValue(index));
    }

    @Override
    public List<BooleanTensor> split(int dimension, long... splitAtIndices) {
        dimension = getAbsoluteDimension(dimension, getRank());

        if (dimension < 0 || dimension >= shape.length) {
            throw new IllegalArgumentException("Invalid dimension to split on " + dimension);
        }

        int[] moveDimToZero = TensorShape.slideDimension(dimension, 0, shape.length);
        int[] moveZeroToDim = TensorShape.slideDimension(0, dimension, shape.length);

        JVMBooleanTensor permutedTensor = (JVMBooleanTensor) this.permute(moveDimToZero);

        BooleanBuffer.PrimitiveBooleanWrapper rawBuffer = permutedTensor.buffer;

        List<BooleanTensor> splitTensor = new ArrayList<>();

        long previousSplitAtIndex = 0;
        int rawBufferPosition = 0;
        for (long splitAtIndex : splitAtIndices) {

            long[] subTensorShape = getShape();
            long subTensorLengthInDimension = splitAtIndex - previousSplitAtIndex;

            if (subTensorLengthInDimension > shape[dimension] || subTensorLengthInDimension <= 0) {
                throw new IllegalArgumentException("Invalid index to split on " + splitAtIndex + " at " + dimension + " for tensor of shape " + Arrays.toString(shape));
            }

            subTensorShape[dimension] = subTensorLengthInDimension;
            int subTensorLength = Ints.checkedCast(TensorShape.getLength(subTensorShape));

            boolean[] buffer = new boolean[subTensorLength];
            System.arraycopy(rawBuffer.asBooleanArray(), rawBufferPosition, buffer, 0, buffer.length);

            long[] subTensorPermutedShape = getPermutedIndices(subTensorShape, moveDimToZero);
            BooleanTensor subTensor = BooleanTensor.create(buffer, subTensorPermutedShape).permute(moveZeroToDim);
            splitTensor.add(subTensor);

            previousSplitAtIndex = splitAtIndex;
            rawBufferPosition += buffer.length;
        }

        return splitTensor;
    }

    @Override
    public BooleanTensor diag() {
        boolean[] newBuffer;
        long[] newShape;
        if (getRank() == 1) {
            int n = buffer.getLength();
            newBuffer = new boolean[Ints.checkedCast((long) n * (long) n)];
            for (int i = 0; i < n; i++) {
                newBuffer[i * n + i] = buffer.get(i);
            }
            newShape = new long[]{n, n};
        } else if (getRank() == 2 && shape[0] == shape[1]) {
            int n = Ints.checkedCast(shape[0]);
            newBuffer = new boolean[n];
            for (int i = 0; i < n; i++) {
                newBuffer[i] = buffer.get(i * n + i);
            }
            newShape = new long[]{n};
        } else {
            throw new IllegalArgumentException("Diag is only valid for vectors or square matrices");
        }

        return new JVMBooleanTensor(newBuffer, newShape);
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
        return Arrays.copyOf(stride, stride.length);
    }

    @Override
    public long getLength() {
        return buffer.getLength();
    }

    @Override
    public BooleanTensor duplicate() {
        return new JVMBooleanTensor(buffer.copy(), copyOf(shape, shape.length), copyOf(stride, stride.length));
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;

        if (o instanceof Tensor) {
            Tensor that = (Tensor) o;
            if (!Arrays.equals(that.getShape(), shape)) return false;
            return Arrays.equals(
                that.asFlatArray(),
                this.asFlatArray()
            );
        }

        return false;
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(buffer);
        result = 31 * result + Arrays.hashCode(shape);
        return result;
    }

    @Override
    public String toString() {

        StringBuilder dataString = new StringBuilder();
        if (buffer.getLength() > 20) {
            dataString.append(Arrays.toString(Arrays.copyOfRange(buffer.asBooleanArray(), 0, 10)));
            dataString.append("...");
            dataString.append(Arrays.toString(Arrays.copyOfRange(buffer.asBooleanArray(), buffer.getLength() - 10, buffer.getLength())));
        } else {
            dataString.append(Arrays.toString(buffer.asBooleanArray()));
        }

        return "{\n" +
            "shape = " + Arrays.toString(shape) +
            "\ndata = " + dataString.toString() +
            "\n}";
    }

    @Override
    public BooleanTensor elementwiseEquals(Boolean value) {
        return Tensor.elementwiseEquals(this, BooleanTensor.create(value, this.getShape()));
    }

    @Override
    public FlattenedView<Boolean> getFlattenedView() {
        return new JVMBooleanFlattenedView();
    }


    private class JVMBooleanFlattenedView implements FlattenedView<Boolean> {

        @Override
        public long size() {
            return buffer.getLength();
        }

        @Override
        public Boolean get(long index) {
            return buffer.get(Ints.checkedCast(index));
        }

        @Override
        public Boolean getOrScalar(long index) {
            if (buffer.getLength() == 1) {
                return get(0);
            } else {
                return get(index);
            }
        }

        @Override
        public void set(long index, Boolean value) {
            buffer.set(value, Ints.checkedCast(index));
        }

    }

    @Override
    public double[] asFlatDoubleArray() {
        return buffer.asDoubleArray();
    }

    @Override
    public int[] asFlatIntegerArray() {
        return buffer.asIntegerArray();
    }

    @Override
    public Boolean[] asFlatArray() {
        return ArrayUtils.toObject(asFlatBooleanArray());
    }

    @Override
    public boolean[] asFlatBooleanArray() {
        return buffer.copy().asBooleanArray();
    }

}
