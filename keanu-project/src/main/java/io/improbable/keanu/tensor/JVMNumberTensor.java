package io.improbable.keanu.tensor;

import io.improbable.keanu.tensor.buffer.JVMBuffer;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import org.apache.commons.lang3.ArrayUtils;

import java.util.function.BiFunction;

import static io.improbable.keanu.tensor.TensorShape.getPermutationForDimensionToDimensionZero;

public abstract class JVMNumberTensor<T extends Number, TENSOR extends NumberTensor<T, TENSOR>, B extends JVMBuffer.PrimitiveArrayWrapper<T, B>> extends JVMTensor<T, TENSOR, B> implements NumberTensor<T, TENSOR> {

    protected JVMNumberTensor(B buffer, long[] shape, long[] stride) {
        super(buffer, shape, stride);
    }

}
