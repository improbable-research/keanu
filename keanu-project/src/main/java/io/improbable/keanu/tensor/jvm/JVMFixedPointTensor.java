package io.improbable.keanu.tensor.jvm;

import io.improbable.keanu.tensor.FixedPointTensor;
import io.improbable.keanu.tensor.jvm.buffer.PrimitiveNumberWrapper;

public abstract class JVMFixedPointTensor<T extends Number, TENSOR extends FixedPointTensor<T, TENSOR>, B extends PrimitiveNumberWrapper<T, B>>
    extends JVMNumberTensor<T, TENSOR, B> implements FixedPointTensor<T, TENSOR> {

    protected JVMFixedPointTensor(B buffer, long[] shape, long[] stride) {
        super(buffer, shape, stride);
    }
}