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

import static io.improbable.keanu.tensor.TensorShape.convertFromFlatIndexToPermutedFlatIndex;
import static io.improbable.keanu.tensor.TensorShape.getAbsoluteDimension;
import static io.improbable.keanu.tensor.TensorShape.getFlatIndex;
import static io.improbable.keanu.tensor.TensorShape.getLengthAsInt;
import static io.improbable.keanu.tensor.TensorShape.getPermutationForDimensionToDimensionZero;
import static io.improbable.keanu.tensor.TensorShape.getPermutedIndices;
import static io.improbable.keanu.tensor.TensorShape.getRowFirstStride;
import static io.improbable.keanu.tensor.TensorShape.getShapeIndices;
import static io.improbable.keanu.tensor.TensorShape.invertedPermute;

public class JVMTensor {

    public static <T, B extends JVMBuffer.PrimitiveArrayWrapper<T>>
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

    public static <T extends Number, B extends JVMBuffer.PrimitiveArrayWrapper<T>>
    IntegerTensor argCompare(JVMBuffer.ArrayWrapperFactory<T, B> factory,
                             B buffer,
                             BiFunction<T, T, Boolean> compareOp,
                             long[] shape, long[] stride, int axis) {

        if (axis >= shape.length) {
            throw new IllegalArgumentException("Cannot take arg max of axis " + axis + " on a " + shape.length + " rank tensor.");
        }

        int[] rearrange = getPermutationForDimensionToDimensionZero(axis, shape);

        B permutedBuffer = permute(factory, buffer, shape, stride, rearrange).outputBuffer;

        int dimLength = (int) (buffer.getLength() / shape[axis]);

        B maxBuffer = factory.createNew(dimLength);
        int[] maxIndex = new int[dimLength];

        for (int i = 0; i < permutedBuffer.getLength(); i++) {

            final int bufferIndex = i % dimLength;
            final T value = permutedBuffer.get(i);

            if (compareOp.apply(value, maxBuffer.get(bufferIndex))) {
                maxBuffer.set(value, bufferIndex);
                maxIndex[bufferIndex] = i / dimLength;
            }

        }

        return IntegerTensor.create(maxIndex, ArrayUtils.remove(shape, axis));
    }

    public static <T, B extends JVMBuffer.PrimitiveArrayWrapper<T>>
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

    public static <T, B extends JVMBuffer.PrimitiveArrayWrapper<T>>
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

    public static <T, B extends JVMBuffer.PrimitiveArrayWrapper<T>>
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

    public static <T, B extends JVMBuffer.PrimitiveArrayWrapper<T>>
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

    public static <T, B extends JVMBuffer.PrimitiveArrayWrapper<T>>
    B concatOnDimensionZero(JVMBuffer.ArrayWrapperFactory<T, B> factory,
                            long[] concatShape, List<B> toConcat) {

        B concatBuffer = factory.createNew(getLengthAsInt(concatShape));
        int bufferPosition = 0;

        for (int i = 0; i < toConcat.size(); i++) {

            JVMBuffer.PrimitiveArrayWrapper<T> cBuffer = toConcat.get(i);

            concatBuffer.copyFrom(cBuffer, 0, bufferPosition, cBuffer.getLength());
            bufferPosition += cBuffer.getLength();
        }

        return concatBuffer;
    }
}
