package io.improbable.keanu.tensor.jvm;

import io.improbable.keanu.tensor.FloatingPointScalarOperations;
import io.improbable.keanu.tensor.FloatingPointTensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.tensor.jvm.buffer.PrimitiveNumberWrapper;

public abstract class JVMFloatingPointTensor<T extends Number, TENSOR extends FloatingPointTensor<T, TENSOR>, B extends PrimitiveNumberWrapper<T, B>>
    extends JVMNumberTensor<T, TENSOR, B> implements FloatingPointTensor<T, TENSOR> {

    protected JVMFloatingPointTensor(B buffer, long[] shape, long[] stride) {
        super(buffer, shape, stride);
    }

    protected abstract FloatingPointScalarOperations<T> getOperations();

    @Override
    public TENSOR sigmoidInPlace() {
        buffer.apply(getOperations()::sigmoid);
        return (TENSOR) this;
    }

    @Override
    public TENSOR reciprocalInPlace() {
        buffer.apply(getOperations()::reciprocal);
        return (TENSOR) this;
    }

    @Override
    public TENSOR sqrtInPlace() {
        buffer.apply(getOperations()::sqrt);
        return (TENSOR) this;
    }

    @Override
    public TENSOR logInPlace() {
        buffer.apply(getOperations()::log);
        return (TENSOR) this;
    }

    @Override
    public TENSOR logGammaInPlace() {
        buffer.apply(getOperations()::logGamma);
        return (TENSOR) this;
    }

    @Override
    public TENSOR digammaInPlace() {
        buffer.apply(getOperations()::digamma);
        return (TENSOR) this;
    }

    @Override
    public TENSOR trigammaInPlace() {
        buffer.apply(getOperations()::trigamma);
        return (TENSOR) this;
    }

    @Override
    public TENSOR sinInPlace() {
        buffer.apply(getOperations()::sin);
        return (TENSOR) this;
    }

    @Override
    public TENSOR cosInPlace() {
        buffer.apply(getOperations()::cos);
        return (TENSOR) this;
    }

    @Override
    public TENSOR tanInPlace() {
        buffer.apply(getOperations()::tan);
        return (TENSOR) this;
    }

    @Override
    public TENSOR atanInPlace() {
        buffer.apply(getOperations()::atan);
        return (TENSOR) this;
    }

    @Override
    public TENSOR atan2InPlace(T y) {
        buffer.applyLeft(getOperations()::atan2, y);
        return (TENSOR) this;
    }

    @Override
    public TENSOR atan2InPlace(TENSOR y) {
        return broadcastableBinaryOpWithAutoBroadcastInPlace((left, right) -> getOperations().atan2(left, right), getAsJVMTensor(y));
    }

    @Override
    public TENSOR asinInPlace() {
        buffer.apply(getOperations()::asin);
        return (TENSOR) this;
    }

    @Override
    public TENSOR acosInPlace() {
        buffer.apply(getOperations()::acos);
        return (TENSOR) this;
    }

    @Override
    public TENSOR sinhInPlace() {
        buffer.apply(getOperations()::sinh);
        return (TENSOR) this;
    }

    @Override
    public TENSOR coshInPlace() {
        buffer.apply(getOperations()::cosh);
        return (TENSOR) this;
    }

    @Override
    public TENSOR tanhInPlace() {
        buffer.apply(getOperations()::tanh);
        return (TENSOR) this;
    }

    @Override
    public TENSOR asinhInPlace() {
        buffer.apply(getOperations()::asinh);
        return (TENSOR) this;
    }

    @Override
    public TENSOR acoshInPlace() {
        buffer.apply(getOperations()::acosh);
        return (TENSOR) this;
    }

    @Override
    public TENSOR atanhInPlace() {
        buffer.apply(getOperations()::atanh);
        return (TENSOR) this;
    }

    @Override
    public TENSOR expInPlace() {
        buffer.apply(getOperations()::exp);
        return (TENSOR) this;
    }

    @Override
    public TENSOR log1pInPlace() {
        buffer.apply(getOperations()::log1p);
        return (TENSOR) this;
    }

    @Override
    public TENSOR log2InPlace() {
        buffer.apply(getOperations()::log2);
        return (TENSOR) this;
    }

    @Override
    public TENSOR log10InPlace() {
        buffer.apply(getOperations()::log10);
        return (TENSOR) this;
    }

    @Override
    public TENSOR exp2InPlace() {
        buffer.apply(getOperations()::exp2);
        return (TENSOR) this;
    }

    @Override
    public TENSOR expM1InPlace() {
        buffer.apply(getOperations()::expM1);
        return (TENSOR) this;
    }

    @Override
    public TENSOR ceilInPlace() {
        buffer.apply(getOperations()::ceil);
        return (TENSOR) this;
    }

    @Override
    public TENSOR floorInPlace() {
        buffer.apply(getOperations()::floor);
        return (TENSOR) this;
    }

    @Override
    public TENSOR roundInPlace() {
        buffer.apply(getOperations()::round);
        return (TENSOR) this;
    }

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

    @Override
    public TENSOR standardizeInPlace() {
        return this.minusInPlace(mean()).divInPlace(standardDeviation());
    }

    @Override
    public IntegerTensor nanArgMax(int axis) {
        final FloatingPointScalarOperations<T> operations = getOperations();
        return argCompare((value, max) -> operations.isNaN(max) || !operations.isNaN(value) && operations.gt(value, max), axis);
    }

    @Override
    public IntegerTensor nanArgMax() {
        final FloatingPointScalarOperations<T> operations = getOperations();
        return IntegerTensor.scalar(argCompare((value, max) -> operations.isNaN(max) || !operations.isNaN(value) && operations.gt(value, max)));
    }

    @Override
    public IntegerTensor nanArgMin(int axis) {
        final FloatingPointScalarOperations<T> operations = getOperations();
        return argCompare((value, min) -> operations.isNaN(min) || !operations.isNaN(value) && operations.lt(value, min), axis);
    }

    @Override
    public IntegerTensor nanArgMin() {
        final FloatingPointScalarOperations<T> operations = getOperations();
        return IntegerTensor.scalar(argCompare((value, min) -> operations.isNaN(min) || !operations.isNaN(value) && operations.lt(value, min)));
    }

    @Override
    public BooleanTensor notNaN() {
        return isApply(getOperations()::notNan);
    }

    @Override
    public BooleanTensor isNaN() {
        return isApply(getOperations()::isNaN);
    }

    @Override
    public BooleanTensor isFinite() {
        return isApply(getOperations()::isFinite);
    }

    @Override
    public BooleanTensor isInfinite() {
        return isApply(getOperations()::isInfinite);
    }

    @Override
    public BooleanTensor isNegativeInfinity() {
        return isApply(getOperations()::isNegativeInfinity);
    }

    @Override
    public BooleanTensor isPositiveInfinity() {
        return isApply(getOperations()::isPositiveInfinity);
    }

}
