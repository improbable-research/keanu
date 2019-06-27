package io.improbable.keanu.tensor;

import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import io.improbable.keanu.tensor.buffer.JVMBuffer;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import org.apache.commons.lang3.ArrayUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.BiFunction;
import java.util.stream.Collectors;

import static io.improbable.keanu.tensor.JVMTensorBroadcast.broadcastIfNeeded;
import static io.improbable.keanu.tensor.TensorShape.convertFromFlatIndexToPermutedFlatIndex;
import static io.improbable.keanu.tensor.TensorShape.getAbsoluteDimension;
import static io.improbable.keanu.tensor.TensorShape.getFlatIndex;
import static io.improbable.keanu.tensor.TensorShape.getLengthAsInt;
import static io.improbable.keanu.tensor.TensorShape.getPermutationForDimensionToDimensionZero;
import static io.improbable.keanu.tensor.TensorShape.getPermutedIndices;
import static io.improbable.keanu.tensor.TensorShape.getReshapeAllowingWildcard;
import static io.improbable.keanu.tensor.TensorShape.getRowFirstStride;
import static io.improbable.keanu.tensor.TensorShape.getShapeIndices;
import static io.improbable.keanu.tensor.TensorShape.invertedPermute;
import static java.util.Arrays.copyOf;

public abstract class JVMTensor<T, TENSOR extends Tensor<T, TENSOR>, B extends JVMBuffer.PrimitiveArrayWrapper<T, B>> implements Tensor<T, TENSOR> {

    protected B buffer;
    protected long[] shape;
    protected long[] stride;

    protected JVMTensor(B buffer, long[] shape, long[] stride) {
        this.buffer = buffer;
        this.shape = shape;
        this.stride = stride;
    }

    @Override
    public int getRank() {
        return shape.length;
    }

    @Override
    public long[] getShape() {
        return copyOf(shape, shape.length);
    }

    @Override
    public long[] getStride() {
        return copyOf(stride, stride.length);
    }

    @Override
    public long getLength() {
        return buffer.getLength();
    }

    @Override
    public TENSOR diag() {
        return createFromResultWrapper(diag(shape.length, shape, buffer, getFactory()));
    }

    public static <T, B extends JVMBuffer.PrimitiveArrayWrapper<T, B>>
    ResultWrapper<T, B> diag(int rank, long[] shape,
                             B buffer, JVMBuffer.ArrayWrapperFactory<T, B> factory) {

        B newBuffer;
        long[] newShape;
        if (rank == 1) {
            int n = buffer.getLength();
            newBuffer = factory.createNew(Ints.checkedCast((long) n * (long) n));
            ;
            for (int i = 0; i < n; i++) {
                newBuffer.set(buffer.get(i), i * n + i);
            }
            newShape = new long[]{n, n};
        } else if (rank == 2 && shape[0] == shape[1]) {
            int n = Ints.checkedCast(shape[0]);
            newBuffer = factory.createNew(n);
            for (int i = 0; i < n; i++) {
                newBuffer.set(buffer.get(i * n + i), i);
            }
            newShape = new long[]{n};
        } else {
            throw new IllegalArgumentException("Diag is only valid for vectors or square matrices");
        }

        return new ResultWrapper<>(newBuffer, newShape, TensorShape.getRowFirstStride(newShape));
    }

    @Override
    public TENSOR permute(int... rearrange) {
        return createFromResultWrapper(permute(getFactory(), buffer, shape, stride, rearrange));
    }

    public static <T, B extends JVMBuffer.PrimitiveArrayWrapper<T, B>>
    ResultWrapper<T, B> permute(JVMBuffer.ArrayWrapperFactory<T, B> factory,
                                B buffer,
                                long[] shape, long[] stride,
                                int... rearrange) {
        Preconditions.checkArgument(rearrange.length == shape.length);

        long[] resultShape = getPermutedIndices(shape, rearrange);
        long[] resultStride = getRowFirstStride(resultShape);
        B newBuffer = factory.createNew(buffer.getLength());

        for (int flatIndex = 0; flatIndex < buffer.getLength(); flatIndex++) {

            int permutedFlatIndex = convertFromFlatIndexToPermutedFlatIndex(
                flatIndex,
                shape, stride,
                resultShape, resultStride,
                rearrange
            );

            newBuffer.set(buffer.get(flatIndex), permutedFlatIndex);
        }

        return new ResultWrapper<>(newBuffer, resultShape, resultStride);
    }

    @Override
    public List<TENSOR> split(int dimension, long... splitAtIndices) {
        return split(getFactory(), buffer, shape, stride, dimension, splitAtIndices).stream()
            .map(this::createFromResultWrapper)
            .collect(Collectors.toList());
    }

    public static <T, B extends JVMBuffer.PrimitiveArrayWrapper<T, B>>
    List<ResultWrapper<T, B>> split(JVMBuffer.ArrayWrapperFactory<T, B> factory,
                                    B fromBuffer, long[] shape, long[] stride, int dimension, long... splitAtIndices) {

        dimension = getAbsoluteDimension(dimension, shape.length);

        if (dimension < 0 || dimension >= shape.length) {
            throw new IllegalArgumentException("Invalid dimension to split on " + dimension);
        }

        int[] moveDimToZero = TensorShape.slideDimension(dimension, 0, shape.length);
        int[] moveZeroToDim = TensorShape.slideDimension(0, dimension, shape.length);

        ResultWrapper<T, B> rawBuffer = permute(factory, fromBuffer, shape, stride, moveDimToZero);

        List<ResultWrapper<T, B>> splitTensor = new ArrayList<>();

        long previousSplitAtIndex = 0;
        int rawBufferPosition = 0;
        for (long splitAtIndex : splitAtIndices) {

            long[] subTensorShape = Arrays.copyOf(shape, shape.length);
            long subTensorLengthInDimension = splitAtIndex - previousSplitAtIndex;

            if (subTensorLengthInDimension > shape[dimension] || subTensorLengthInDimension <= 0) {
                throw new IllegalArgumentException("Invalid index to split on " + splitAtIndex + " at " + dimension + " for tensor of shape " + Arrays.toString(shape));
            }

            subTensorShape[dimension] = subTensorLengthInDimension;
            int subTensorLength = Ints.checkedCast(TensorShape.getLength(subTensorShape));

            B buffer = factory.createNew(subTensorLength);
            buffer.copyFrom(rawBuffer.outputBuffer, rawBufferPosition, 0, subTensorLength);

            long[] subTensorPermutedShape = getPermutedIndices(subTensorShape, moveDimToZero);

            ResultWrapper<T, B> result = permute(factory, buffer, subTensorPermutedShape, getRowFirstStride(subTensorPermutedShape), moveZeroToDim);

            splitTensor.add(result);

            previousSplitAtIndex = splitAtIndex;
            rawBufferPosition += buffer.getLength();
        }

        return splitTensor;
    }

    @Override
    public TENSOR slice(int dimension, long index) {
        return createFromResultWrapper(slice(getFactory(), buffer, shape, stride, dimension, index));
    }

    public static <T, B extends JVMBuffer.PrimitiveArrayWrapper<T, B>>
    ResultWrapper<T, B> slice(JVMBuffer.ArrayWrapperFactory<T, B> factory,
                              B buffer,
                              long[] shape, long[] stride,
                              int dimension, long index) {

        Preconditions.checkArgument(dimension < shape.length && index < shape[dimension]);
        long[] resultShape = ArrayUtils.remove(shape, dimension);
        long[] resultStride = getRowFirstStride(resultShape);
        B newBuffer = factory.createNew(getLengthAsInt(resultShape));

        for (int i = 0; i < newBuffer.getLength(); i++) {

            long[] shapeIndices = ArrayUtils.insert(dimension, getShapeIndices(resultShape, resultStride, i), index);

            int j = Ints.checkedCast(getFlatIndex(shape, stride, shapeIndices));

            newBuffer.set(buffer.get(j), i);
        }

        return new ResultWrapper<>(newBuffer, resultShape, resultStride);
    }

    public static <T, B extends JVMBuffer.PrimitiveArrayWrapper<T, B>>
    ResultWrapper<T, B> concat(JVMBuffer.ArrayWrapperFactory<T, B> factory,
                               Tensor[] tensors,
                               int dimension, List<B> toConcat) {

        long[] concatShape = TensorShape.getConcatResultShape(dimension, tensors);

        boolean shouldRearrange = dimension != 0;

        if (shouldRearrange) {

            int[] rearrange = getPermutationForDimensionToDimensionZero(dimension, concatShape);

            List<B> toConcatOnDimensionZero = new ArrayList<>();

            for (int i = 0; i < toConcat.size(); i++) {
                toConcatOnDimensionZero.add(permute(factory, toConcat.get(i), tensors[i].getShape(), tensors[i].getStride(), rearrange).outputBuffer);
            }

            long[] permutedConcatShape = getPermutedIndices(concatShape, rearrange);

            B concatOnDimZero = concatOnDimensionZero(factory, permutedConcatShape, toConcatOnDimensionZero);

            return permute(factory, concatOnDimZero, permutedConcatShape, getRowFirstStride(permutedConcatShape), invertedPermute(rearrange));

        } else {

            B buffer = concatOnDimensionZero(factory, concatShape, toConcat);
            return new ResultWrapper<>(buffer, concatShape, getRowFirstStride(concatShape));
        }
    }

    public static <T, B extends JVMBuffer.PrimitiveArrayWrapper<T, B>>
    B concatOnDimensionZero(JVMBuffer.ArrayWrapperFactory<T, B> factory,
                            long[] concatShape, List<B> toConcat) {

        B concatBuffer = factory.createNew(getLengthAsInt(concatShape));
        int bufferPosition = 0;

        for (int i = 0; i < toConcat.size(); i++) {

            JVMBuffer.PrimitiveArrayWrapper<T, B> cBuffer = toConcat.get(i);

            concatBuffer.copyFrom(cBuffer, 0, bufferPosition, cBuffer.getLength());
            bufferPosition += cBuffer.getLength();
        }

        return concatBuffer;
    }

    protected TENSOR broadcastableBinaryOpWithAutoBroadcast(BiFunction<T, T, T> op,
                                                            JVMTensor<T, TENSOR, B> right) {

        final ResultWrapper<T, B> result = broadcastIfNeeded(
            getFactory(),
            buffer, shape, stride, buffer.getLength(),
            right.buffer, right.shape, right.stride, right.buffer.getLength(),
            op, true
        );

        return set(result.outputBuffer, result.outputShape, result.outputStride);
    }

    @Override
    public TENSOR reshape(long... newShape) {
        long[] normalizedShape = getReshapeAllowingWildcard(shape, buffer.getLength(), newShape);
        return create(buffer.copy(), normalizedShape, getRowFirstStride(normalizedShape));
    }

    @Override
    public TENSOR broadcast(long... toShape) {
        int outputLength = TensorShape.getLengthAsInt(toShape);
        long[] outputStride = TensorShape.getRowFirstStride(toShape);
        B outputBuffer = getFactory().createNew(outputLength);

        JVMTensorBroadcast.broadcast(buffer, shape, stride, outputBuffer, outputStride);

        return create(outputBuffer, toShape, outputStride);
    }

    public IntegerTensor argCompare(BiFunction<T, T, Boolean> compareOp, int axis) {
        return argCompare(getFactory(), buffer, compareOp, shape, stride, axis);
    }

    public static <T, B extends JVMBuffer.PrimitiveArrayWrapper<T, B>>
    IntegerTensor argCompare(JVMBuffer.ArrayWrapperFactory<T, B> factory,
                             B buffer,
                             BiFunction<T, T, Boolean> compareOp,
                             long[] shape, long[] stride, int axis) {

        if (axis >= shape.length) {
            throw new IllegalArgumentException("Cannot take arg max of axis " + axis + " on a " + shape.length + " rank tensor.");
        }

        int[] rearrange = getPermutationForDimensionToDimensionZero(axis, shape);

        B permutedBuffer = JVMTensor.permute(factory, buffer, shape, stride, rearrange).outputBuffer;

        int dimLength = (int) (buffer.getLength() / shape[axis]);

        B maxBuffer = factory.createNew(dimLength);
        int[] maxIndex = new int[dimLength];
        Arrays.fill(maxIndex, -1);

        for (int i = 0; i < permutedBuffer.getLength(); i++) {

            final int bufferIndex = i % dimLength;
            final T value = permutedBuffer.get(i);

            if (maxIndex[bufferIndex] < 0 || compareOp.apply(value, maxBuffer.get(bufferIndex))) {
                maxBuffer.set(value, bufferIndex);
                maxIndex[bufferIndex] = i / dimLength;
            }

        }

        return IntegerTensor.create(maxIndex, ArrayUtils.remove(shape, axis));
    }

    public int argCompare(BiFunction<T, T, Boolean> compareOp) {
        return argCompare(buffer, compareOp);
    }

    public static <T, B extends JVMBuffer.PrimitiveArrayWrapper<T, B>> int argCompare(B buffer,
                                                                                      BiFunction<T, T, Boolean> compareOp) {
        T min = null;
        int argMin = -1;
        for (int i = 0; i < buffer.getLength(); i++) {

            final T value = buffer.get(i);
            if (argMin < 0 || compareOp.apply(value, min)) {
                min = value;
                argMin = i;
            }
        }

        return argMin;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        JVMTensor that = (JVMTensor) o;
        return Arrays.equals(shape, that.shape) && buffer.equals(that.buffer);
    }

    @Override
    public String toString() {

        StringBuilder dataString = new StringBuilder();
        if (buffer.getLength() > 20) {
            dataString.append(Arrays.toString(Arrays.copyOfRange(buffer.asArray(), 0, 10)));
            dataString.append("...");
            dataString.append(Arrays.toString(Arrays.copyOfRange(buffer.asArray(), buffer.getLength() - 10, buffer.getLength())));
        } else {
            dataString.append(Arrays.toString(buffer.asArray()));
        }

        return "{\n" +
            "shape = " + Arrays.toString(shape) +
            "\ndata = " + dataString.toString() +
            "\n}";
    }

    @Override
    public int hashCode() {
        int result = Arrays.hashCode(shape);
        result = 31 * result + buffer.hashCode();
        return result;
    }

    private TENSOR createFromResultWrapper(ResultWrapper<T, B> wrapper) {
        return create(wrapper.outputBuffer, wrapper.outputShape, wrapper.outputStride);
    }

    protected abstract TENSOR create(B buffer, long[] shape, long[] stride);

    protected abstract TENSOR set(B buffer, long[] shape, long[] stride);

    protected abstract JVMBuffer.ArrayWrapperFactory<T, B> getFactory();
}
