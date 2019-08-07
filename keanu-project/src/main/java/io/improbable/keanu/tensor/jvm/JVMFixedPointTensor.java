package io.improbable.keanu.tensor.jvm;

import io.improbable.keanu.tensor.FixedPointScalarOperations;
import io.improbable.keanu.tensor.FixedPointTensor;
import io.improbable.keanu.tensor.jvm.buffer.PrimitiveNumberWrapper;

public abstract class JVMFixedPointTensor<T extends Number, TENSOR extends FixedPointTensor<T, TENSOR>, B extends PrimitiveNumberWrapper<T, B>>
    extends JVMNumberTensor<T, TENSOR, B> implements FixedPointTensor<T, TENSOR> {

    protected JVMFixedPointTensor(B buffer, long[] shape, long[] stride) {
        super(buffer, shape, stride);
    }

    @Override
    public TENSOR modInPlace(T that) {
        final FixedPointScalarOperations<T> operations = getOperations();
        buffer.apply(v -> operations.mod(v, that));
        return (TENSOR) this;
    }

    @Override
    public TENSOR modInPlace(TENSOR that) {
        return broadcastableBinaryOpWithAutoBroadcastInPlace(getOperations()::mod, getAsJVMTensor(that));
    }

    @Override
    protected abstract FixedPointScalarOperations<T> getOperations();
}