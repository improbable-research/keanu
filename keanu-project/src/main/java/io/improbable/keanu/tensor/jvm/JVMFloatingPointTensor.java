package io.improbable.keanu.tensor.jvm;

import io.improbable.keanu.tensor.FloatingPointTensor;
import io.improbable.keanu.tensor.buffer.PrimitiveNumberWrapper;

public abstract class JVMFloatingPointTensor<T extends Number, TENSOR extends FloatingPointTensor<T, TENSOR>, B extends PrimitiveNumberWrapper<T, B>>
    extends JVMNumberTensor<T, TENSOR, B> implements FloatingPointTensor<T, TENSOR> {

    protected JVMFloatingPointTensor(B buffer, long[] shape, long[] stride) {
        super(buffer, shape, stride);
    }
}
