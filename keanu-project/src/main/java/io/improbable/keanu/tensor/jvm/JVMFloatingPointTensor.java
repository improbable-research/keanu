package io.improbable.keanu.tensor.jvm;

import io.improbable.keanu.tensor.FloatingPointScalarOperations;
import io.improbable.keanu.tensor.FloatingPointTensor;
import io.improbable.keanu.tensor.jvm.buffer.PrimitiveNumberWrapper;

public abstract class JVMFloatingPointTensor<T extends Number, TENSOR extends FloatingPointTensor<T, TENSOR>, B extends PrimitiveNumberWrapper<T, B>>
    extends JVMNumberTensor<T, TENSOR, B> implements FloatingPointTensor<T, TENSOR> {

    protected JVMFloatingPointTensor(B buffer, long[] shape, long[] stride) {
        super(buffer, shape, stride);
    }

    protected abstract FloatingPointScalarOperations<T> getOperations();

    @Override
    public TENSOR safeLogTimesInPlace(TENSOR y) {
        return broadcastableBinaryOpWithAutoBroadcastInPlace(getOperations()::safeLogTimes, getAsJVMTensor(y));
    }

    @Override
    public TENSOR logAddExp2InPlace(TENSOR that) {
        return broadcastableBinaryOpWithAutoBroadcastInPlace(getOperations()::logAddExp2, getAsJVMTensor(that));
    }

    @Override
    public TENSOR logAddExpInPlace(TENSOR that) {
        return broadcastableBinaryOpWithAutoBroadcastInPlace(getOperations()::logAddExp, getAsJVMTensor(that));
    }
}
