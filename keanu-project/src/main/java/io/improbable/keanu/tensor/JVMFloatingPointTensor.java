package io.improbable.keanu.tensor;

import io.improbable.keanu.tensor.buffer.JVMBuffer;

public abstract class JVMFloatingPointTensor<T extends Number, TENSOR extends FloatingPointTensor<T, TENSOR>, B extends JVMBuffer.PrimitiveArrayWrapper<T>> extends JVMNumberTensor<T, TENSOR, B> implements FloatingPointTensor<T, TENSOR> {

    protected JVMFloatingPointTensor(B buffer, long[] shape, long[] stride) {
        super(buffer, shape, stride);
    }
}
