package io.improbable.keanu.tensor;

import io.improbable.keanu.tensor.buffer.JVMBuffer;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import org.apache.commons.lang3.ArrayUtils;

import java.util.function.BiFunction;

import static io.improbable.keanu.tensor.TensorShape.getPermutationForDimensionToDimensionZero;

public abstract class JVMNumberTensor<T extends Number, TENSOR extends NumberTensor<T, TENSOR>, B extends JVMBuffer.PrimitiveArrayWrapper<T>> extends JVMTensor<T, TENSOR, B> implements NumberTensor<T, TENSOR> {

    protected JVMNumberTensor(B buffer, long[] shape, long[] stride) {
        super(buffer, shape, stride);
    }

    public IntegerTensor argCompare(BiFunction<T, T, Boolean> compareOp, int axis) {
        return argCompare(getFactory(), buffer, compareOp, shape, stride, axis);
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

        B permutedBuffer = JVMTensor.permute(factory, buffer, shape, stride, rearrange).outputBuffer;

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
}
