package io.improbable.keanu.tensor.jvm;

import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.bool.BooleanBuffer;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.bool.JVMBooleanTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.tensor.jvm.buffer.JVMBuffer;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.StringUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.Collectors;

import static io.improbable.keanu.tensor.TensorShape.convertFromFlatIndexToPermutedFlatIndex;
import static io.improbable.keanu.tensor.TensorShape.getAbsoluteDimension;
import static io.improbable.keanu.tensor.TensorShape.getBroadcastResultShape;
import static io.improbable.keanu.tensor.TensorShape.getPermutationForDimensionToDimensionZero;
import static io.improbable.keanu.tensor.TensorShape.getPermutedIndices;
import static io.improbable.keanu.tensor.TensorShape.getReshapeAllowingWildcard;
import static io.improbable.keanu.tensor.TensorShape.getRowFirstStride;
import static io.improbable.keanu.tensor.TensorShape.incrementIndexByShape;
import static io.improbable.keanu.tensor.TensorShape.invertedPermute;
import static io.improbable.keanu.tensor.jvm.JVMTensorBroadcast.broadcastIfNeeded;
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

    public B getBuffer() {
        return buffer;
    }

    protected abstract JVMTensor<T, TENSOR, B> getAsJVMTensor(TENSOR that);

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
    public TENSOR duplicate() {
        return create(buffer.copy(), Arrays.copyOf(shape, shape.length), Arrays.copyOf(stride, stride.length));
    }

    @Override
    public TENSOR get(BooleanTensor booleanIndex) {

        List<Long> indices = new ArrayList<>();

        FlattenedView<Boolean> flattenedView = booleanIndex.getFlattenedView();
        for (long i = 0; i < booleanIndex.getLength(); i++) {
            if (flattenedView.get(i)) {
                indices.add(i);
            }
        }

        B newBuffer = getFactory().createNew(indices.size());

        for (int i = 0; i < newBuffer.getLength(); i++) {
            newBuffer.set(buffer.get(indices.get(i)), i);
        }

        return create(newBuffer, new long[]{newBuffer.getLength()}, new long[]{1});
    }

    @Override
    public TENSOR diagPart() {
        return createFromResultWrapper(diagPart(shape, buffer, getFactory()));
    }

    public static <T, B extends JVMBuffer.PrimitiveArrayWrapper<T, B>>
    ResultWrapper<T, B> diagPart(long[] shape,
                                 B buffer, JVMBuffer.ArrayWrapperFactory<T, B> factory) {

        Preconditions.checkArgument(shape.length >= 2, "Diag Part only operates on matrices or greater rank");

        final long N = shape[shape.length - 1];
        final long[] resultShape = TensorShape.getDiagPartResultShape(shape);
        final long bufferLength = TensorShape.getLength(resultShape);
        final B newBuffer = factory.createNew(bufferLength);

        for (long i = 0; i < bufferLength; i++) {
            final long pos = i + N * i - (i / N) * N;
            newBuffer.set(buffer.get(pos), i);
        }

        return new ResultWrapper<>(newBuffer, resultShape, TensorShape.getRowFirstStride(resultShape));
    }

    @Override
    public TENSOR diag() {
        return createFromResultWrapper(diag(shape, buffer, getFactory()));
    }

    public static <T, B extends JVMBuffer.PrimitiveArrayWrapper<T, B>>
    ResultWrapper<T, B> diag(long[] shape,
                             B buffer, JVMBuffer.ArrayWrapperFactory<T, B> factory) {

        Preconditions.checkArgument(shape.length >= 1, "Diag operates on rank >= 1");

        final long endDim = shape[shape.length - 1];
        final long bufferLength = buffer.getLength();
        final B newBuffer = factory.createNew(bufferLength * endDim);

        for (long i = 0; i < bufferLength; i++) {
            final long pos = i + endDim * i - (i / endDim) * endDim;
            newBuffer.set(buffer.get(i), pos);
        }

        final long[] newShape = TensorShape.getDiagResultShape(shape);
        return new ResultWrapper<>(newBuffer, newShape, TensorShape.getRowFirstStride(newShape));
    }

    @Override
    public TENSOR triUpper(int k) {
        Preconditions.checkArgument(shape.length >= 2, "Tri Upper input must be rank >= 2");

        long N = shape[shape.length - 2];
        long M = shape[shape.length - 1];
        long[] batchShape = ArrayUtils.subarray(shape, 0, shape.length - 2);
        long batchLength = TensorShape.getLength(batchShape);
        long batchSize = N * M;

        B toBuffer = getFactory().createNew(batchLength * batchSize);

        for (int i = 0; i < batchLength; i++) {
            copyUpperTriangle(N, M, toBuffer, i * batchSize, buffer, k);
        }

        return create(toBuffer, getShape(), getStride());
    }

    private void copyUpperTriangle(long N, long M, B to, long toOffset, B from, int k) {
        for (long i = 0; i < N; i++) {
            for (long j = Math.max(0, i + k); j < M; j++) {
                final long index = toOffset + i * M + j;
                to.set(from.get(index), index);
            }
        }
    }

    @Override
    public TENSOR triLower(int k) {
        Preconditions.checkArgument(shape.length >= 2, "Tri Lower input must be rank >= 2");

        long N = shape[shape.length - 2];
        long M = shape[shape.length - 1];
        long[] batchShape = ArrayUtils.subarray(shape, 0, shape.length - 2);
        long batchLength = TensorShape.getLength(batchShape);
        long batchSize = N * M;

        B toBuffer = getFactory().createNew(batchLength * batchSize);

        for (int i = 0; i < batchLength; i++) {
            copyLowerTriangle(N, M, toBuffer, i * batchSize, buffer, k);
        }

        return create(toBuffer, getShape(), getStride());
    }

    private void copyLowerTriangle(long N, long M, B to, long toOffset, B from, int k) {
        for (long i = 0; i < N; i++) {
            for (long j = 0; j < Math.min(M, i - k + 1); j++) {
                final long index = toOffset + i * M + j;
                to.set(from.get(index), index);
            }
        }
    }

    @Override
    public TENSOR fillTriangular(boolean fillUpper, boolean fillLower) {
        Preconditions.checkArgument(shape.length >= 1, "Fill symmetric works on rank >= 1");

        final long endDim = shape[shape.length - 1];
        double a = Math.sqrt(1 + 8 * endDim);

        Preconditions.checkArgument(
            a == Math.floor(a),
            "Length " + endDim + " is not the correct number of elements for a triangular matrix"
        );

        final long N = ((long) a - 1) / 2;
        final long NN = N * N;
        long[] resultShape = TensorShape.concat(ArrayUtils.subarray(shape, 0, shape.length - 1), new long[]{N, N});
        final B newBuffer = getFactory().createNew(TensorShape.getLength(resultShape));

        long row = 0;
        long col = 0;
        final long bufferLength = buffer.getLength();
        for (long i = 0; i < bufferLength; i++) {

            final long batchNum = i / endDim;
            final long batchOffset = batchNum * NN;

            if (i % endDim == 0) {
                row = 0;
                col = 0;
            }

            final long upperPos = batchOffset + row * N + col;
            if (fillUpper) {
                newBuffer.set(buffer.get(i), upperPos);
            }

            if (fillLower) {
                final long lowerPos = upperPos + (col - row) * (N - 1);
                newBuffer.set(buffer.get(i), lowerPos);
            }

            col++;
            if (col == N) {
                row++;
                col = row;
            }
        }

        return create(newBuffer, resultShape, TensorShape.getRowFirstStride(resultShape));
    }

    @Override
    public TENSOR trianglePart(boolean upperPart) {
        final long N = shape[shape.length - 2];
        final long M = shape[shape.length - 1];
        Preconditions.checkArgument(N == M, "Triangle part only supports square matrices");

        final long[] batchShape = ArrayUtils.subarray(shape, 0, shape.length - 2);
        final long batchLength = TensorShape.getLength(batchShape);
        final long batchSize = N * M;

        final long vectorLength = N * (N + 1) / 2;
        final long[] resultShape = TensorShape.concat(batchShape, new long[]{vectorLength});
        final long resultLength = TensorShape.getLength(resultShape);
        B resultBuffer = getFactory().createNew(resultLength);

        int pos = 0;
        if(upperPart) {
            for (int i = 0; i < batchLength; i++) {
                long batchOffset = i * batchSize;
                for (int r = 0; r < N; r++) {
                    for (int c = r; c < N; c++) {
                        resultBuffer.set(buffer.get(batchOffset + r * N + c), pos);
                        pos++;
                    }
                }
            }
        }else{
            for (int i = 0; i < batchLength; i++) {
                long batchOffset = i * batchSize;
                for (int c = 0; c < N; c++) {
                    for (int r = c; r < N; r++) {
                        resultBuffer.set(buffer.get(batchOffset + r * N + c), pos);
                        pos++;
                    }
                }
            }
        }

        return create(resultBuffer, resultShape, TensorShape.getRowFirstStride(resultShape));
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

        for (long flatIndex = 0; flatIndex < buffer.getLength(); flatIndex++) {

            long permutedFlatIndex = convertFromFlatIndexToPermutedFlatIndex(
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
        long rawBufferPosition = 0;
        for (long splitAtIndex : splitAtIndices) {

            long[] subTensorShape = Arrays.copyOf(shape, shape.length);
            long subTensorLengthInDimension = splitAtIndex - previousSplitAtIndex;

            if (subTensorLengthInDimension > shape[dimension] || subTensorLengthInDimension <= 0) {
                throw new IllegalArgumentException("Invalid index to split on " + splitAtIndex + " at " + dimension + " for tensor of shape " + Arrays.toString(shape));
            }

            subTensorShape[dimension] = subTensorLengthInDimension;
            long subTensorLength = TensorShape.getLength(subTensorShape);

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
    public TENSOR take(long... index) {
        B newBuffer = getFactory().createNew(1);
        newBuffer.set(buffer.get(TensorShape.getFlatIndex(shape, stride, index)), 0);
        return create(newBuffer, new long[0], new long[]{1});
    }

    @Override
    public TENSOR slice(int dimension, long index) {
        return createFromResultWrapper(slice(getFactory(), buffer, new DimensionIndexMapper(shape, stride, dimension, index)));
    }

    @Override
    public TENSOR slice(Slicer slicer) {
        return createFromResultWrapper(slice(getFactory(), buffer, new SlicerIndexMapper(slicer, shape, stride)));
    }

    public static <T, B extends JVMBuffer.PrimitiveArrayWrapper<T, B>>
    ResultWrapper<T, B> slice(JVMBuffer.ArrayWrapperFactory<T, B> factory,
                              B buffer,
                              IndexMapper indexMapper) {

        final long[] resultShape = indexMapper.getResultShape();
        final long[] resultStride = indexMapper.getResultStride();
        B newBuffer = factory.createNew(TensorShape.getLength(resultShape));

        for (long i = 0; i < newBuffer.getLength(); i++) {

            final long j = indexMapper.getSourceIndexFromResultIndex(i);

            newBuffer.set(buffer.get(j), i);
        }

        return new ResultWrapper<>(newBuffer, resultShape, resultStride);
    }

    @Override
    public TENSOR reverseSlice(TENSOR setTo, Slicer slicer) {

        JVMTensor.reverseSlice(
            getFlattenedView(),
            setTo.getFlattenedView(),
            new SlicerIndexMapper(slicer, setTo.getShape(), setTo.getStride())
        );

        return setTo;
    }

    public static <T> void reverseSlice(FlattenedView<T> from,
                                        FlattenedView<T> to,
                                        IndexMapper indexMapper) {

        for (long i = 0; i < from.size(); i++) {

            final long j = indexMapper.getSourceIndexFromResultIndex(i);
            to.set(j, from.get(i));
        }

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

        B concatBuffer = factory.createNew(TensorShape.getLength(concatShape));
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
            op, false
        );

        return create(result.outputBuffer, result.outputShape, result.outputStride);
    }

    protected TENSOR broadcastableBinaryOpWithAutoBroadcastInPlace(BiFunction<T, T, T> op,
                                                                   JVMTensor<T, TENSOR, B> right) {

        final ResultWrapper<T, B> result = broadcastIfNeeded(
            getFactory(),
            buffer, shape, stride, buffer.getLength(),
            right.buffer, right.shape, right.stride, right.buffer.getLength(),
            op, true
        );

        return set(result.outputBuffer, result.outputShape, result.outputStride);
    }

    protected BooleanTensor broadcastableBinaryOpToBooleanWithAutoBroadcast(BiFunction<T, T, Boolean> op,
                                                                            JVMTensor<T, TENSOR, B> right) {


        final ResultWrapper<Boolean, BooleanBuffer.PrimitiveBooleanWrapper> result = broadcastIfNeeded(
            BooleanBuffer.factory,
            buffer, shape, stride, buffer.getLength(),
            right.buffer, right.shape, right.stride, right.buffer.getLength(),
            op, false
        );

        return new JVMBooleanTensor(result.outputBuffer, result.outputShape, result.outputStride);
    }

    @Override
    public TENSOR reshape(long... newShape) {
        long[] normalizedShape = getReshapeAllowingWildcard(shape, buffer.getLength(), newShape);
        return create(buffer.copy(), normalizedShape, getRowFirstStride(normalizedShape));
    }

    @Override
    public TENSOR broadcast(long... toShape) {
        long outputLength = TensorShape.getLength(toShape);
        long[] outputStride = TensorShape.getRowFirstStride(toShape);
        B outputBuffer = getFactory().createNew(outputLength);

        TensorShape.getBroadcastResultShape(shape, toShape);

        JVMTensorBroadcast.broadcast(buffer, shape, stride, outputBuffer, outputStride);

        return create(outputBuffer, toShape, outputStride);
    }

    @Override
    public TENSOR where(BooleanTensor predicate, TENSOR els) {

        final long[] resultShape = getBroadcastResultShape(getBroadcastResultShape(shape, predicate.getShape()), els.getShape());
        final TENSOR broadcastedTrue = this.hasShape(resultShape) ? (TENSOR) this : this.broadcast(resultShape);
        final TENSOR broadcastedFalse = els.hasShape(resultShape) ? els : els.broadcast(resultShape);
        final BooleanTensor broadcastedPredicate = predicate.hasShape(resultShape) ? predicate : predicate.broadcast(resultShape);

        FlattenedView<T> trueValuesFlattened = broadcastedTrue.getFlattenedView();
        FlattenedView<T> falseValuesFlattened = broadcastedFalse.getFlattenedView();
        FlattenedView<Boolean> predicateValuesFlattened = broadcastedPredicate.getFlattenedView();

        B newBuffer = getFactory().createNew(TensorShape.getLengthAsInt(resultShape));
        for (int i = 0; i < newBuffer.getLength(); i++) {
            final T value = predicateValuesFlattened.get(i) ? trueValuesFlattened.get(i) : falseValuesFlattened.get(i);
            newBuffer.set(value, i);
        }

        final long[] resultStride = broadcastedTrue.getStride();
        return create(newBuffer, copyOf(resultShape, resultShape.length), copyOf(resultStride, resultStride.length));
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
            if (i == 0 || compareOp.apply(value, min)) {
                min = value;
                argMin = i;
            }
        }

        return argMin;
    }

    public BooleanTensor isApply(Function<T, Boolean> op) {
        boolean[] newBuffer = new boolean[Ints.checkedCast(buffer.getLength())];

        for (int i = 0; i < buffer.getLength(); i++) {
            newBuffer[i] = op.apply(buffer.get(i));
        }

        return BooleanTensor.create(newBuffer, Arrays.copyOf(shape, shape.length));
    }

    @Override
    public FlattenedView<T> getFlattenedView() {
        if (buffer.getLength() == 1) {
            return new ScalarJVMFlattenedView();
        } else {
            return new TensorJVMFlattenedView();
        }
    }

    private class JVMFlattenedView {
        public long size() {
            return buffer.getLength();
        }

        public T get(long index) {
            return buffer.get(index);
        }

        public void set(long index, T value) {
            buffer.set(value, index);
        }
    }

    private class TensorJVMFlattenedView extends JVMFlattenedView implements FlattenedView<T> {
        @Override
        public T getOrScalar(long index) {
            return get(index);
        }
    }

    private class ScalarJVMFlattenedView extends JVMFlattenedView implements FlattenedView<T> {
        @Override
        public T getOrScalar(long index) {
            return buffer.get(0);
        }
    }

    @Override
    public BooleanTensor elementwiseEquals(T value) {
        return new JVMBooleanTensor(buffer.equal(value), Arrays.copyOf(shape, shape.length), Arrays.copyOf(stride, stride.length));
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

        return "{\n" +
            "shape = " + Arrays.toString(shape) +
            "\ndata = \n" + arrayToString(buffer.asArray(), shape, stride) +
            "\n}";
    }

    private String arrayToString(Object[] array, long[] shape, long[] stride) {

        if (shape.length == 0) {
            return array[0].toString();
        }

        StringBuilder sb = new StringBuilder();
        sb.append("[");
        arrayToString(sb, array, 0, shape, stride, new long[shape.length]);
        sb.append("]");
        return sb.toString();
    }

    private void arrayToString(StringBuilder sb, Object[] array, int dimension, long[] shape, long[] stride,
                               long[] index) {

        if (dimension >= shape.length - 1) {
            for (int i = 0; i < shape[dimension]; i++) {
                sb.append(array[Ints.checkedCast(TensorShape.getFlatIndex(shape, stride, index))]);
                incrementIndexByShape(shape, index);

                if (i + 1 < shape[dimension]) {
                    sb.append(", ");
                }
            }
        } else {
            for (int i = 0; i < shape[dimension]; i++) {
                sb.append("[");
                arrayToString(sb, array, dimension + 1, shape, stride, index);

                sb.append("]");

                if (i + 1 < shape[dimension]) {
                    sb.append(",");
                    sb.append(StringUtils.repeat("\n", shape.length - 1 - dimension));
                    sb.append(StringUtils.repeat(" ", dimension + 1));
                }
            }
        }
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
