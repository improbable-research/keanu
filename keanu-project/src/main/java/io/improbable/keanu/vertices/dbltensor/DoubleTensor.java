package io.improbable.keanu.vertices.dbltensor;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

public interface DoubleTensor extends Tensor {

    static DoubleTensor create(double value, int[] shape) {
        if (Arrays.equals(shape, Tensor.SCALAR_SHAPE)) {
            return new SimpleDoubleTensor(value);
        } else {
            return Nd4jDoubleTensor.create(value, shape);
        }
    }

    static DoubleTensor create(double[] values, int[] shape) {
        if (Arrays.equals(shape, Tensor.SCALAR_SHAPE) && values.length == 1) {
            return new SimpleDoubleTensor(values[0]);
        } else {
            return Nd4jDoubleTensor.create(values, shape);
        }
    }

    static DoubleTensor ones(int[] shape) {
        if (Arrays.equals(shape, Tensor.SCALAR_SHAPE)) {
            return new SimpleDoubleTensor(1.0);
        } else {
            return Nd4jDoubleTensor.ones(shape);
        }
    }

    static DoubleTensor zeros(int[] shape) {
        if (Arrays.equals(shape, Tensor.SCALAR_SHAPE)) {
            return new SimpleDoubleTensor(0.0);
        } else {
            return Nd4jDoubleTensor.zeros(shape);
        }
    }

    static DoubleTensor scalar(double scalarValue) {
        return new SimpleDoubleTensor(scalarValue);
    }

    static DoubleTensor placeHolder(int[] shape) {
        return new SimpleDoubleTensor(shape);
    }

    static Map<Long, DoubleTensor> fromScalars(Map<Long, Double> scalars) {
        Map<Long, DoubleTensor> asTensors = new HashMap<>();

        for (Map.Entry<Long, Double> entry : scalars.entrySet()) {
            asTensors.put(entry.getKey(), DoubleTensor.scalar(entry.getValue()));
        }

        return asTensors;
    }

    static Map<Long, Double> toScalars(Map<Long, DoubleTensor> tensors) {
        Map<Long, Double> asScalars = new HashMap<>();

        for (Map.Entry<Long, DoubleTensor> entry : tensors.entrySet()) {
            asScalars.put(entry.getKey(), entry.getValue().scalar());
        }

        return asScalars;
    }

    double getValue(int... index);

    void setValue(double value, int... index);

    double scalar();

    double sum();

    DoubleTensor duplicate();

    //New tensor Ops and transforms

    DoubleTensor reciprocal();

    DoubleTensor minus(double value);

    DoubleTensor plus(double value);

    DoubleTensor times(double value);

    DoubleTensor div(double value);

    DoubleTensor pow(DoubleTensor exponent);

    DoubleTensor pow(double exponent);

    DoubleTensor sqrt();

    DoubleTensor log();

    DoubleTensor sin();

    DoubleTensor cos();

    DoubleTensor asin();

    DoubleTensor acos();

    DoubleTensor exp();

    DoubleTensor minus(DoubleTensor that);

    DoubleTensor plus(DoubleTensor that);

    DoubleTensor times(DoubleTensor that);

    DoubleTensor div(DoubleTensor that);

    DoubleTensor unaryMinus();

    DoubleTensor getGreaterThanMask(DoubleTensor greaterThanThis);

    DoubleTensor getGreaterThanOrEqualToMask(DoubleTensor greaterThanThis);

    DoubleTensor getLessThanMask(DoubleTensor lessThanThis);

    DoubleTensor getLessThanOrEqualToMask(DoubleTensor lessThanThis);

    DoubleTensor setWithMaskInPlace(DoubleTensor mask, double value);

    DoubleTensor setWithMask(DoubleTensor mask, double value);

    DoubleTensor abs();

    DoubleTensor apply(Function<Double, Double> function);

    //In place Ops and Transforms. These mutate the source vertex (i.e. this).

    DoubleTensor reciprocalInPlace();

    DoubleTensor minusInPlace(double value);

    DoubleTensor plusInPlace(double value);

    DoubleTensor timesInPlace(double value);

    DoubleTensor divInPlace(double value);

    DoubleTensor powInPlace(DoubleTensor exponent);

    DoubleTensor powInPlace(double exponent);

    DoubleTensor sqrtInPlace();

    DoubleTensor logInPlace();

    DoubleTensor sinInPlace();

    DoubleTensor cosInPlace();

    DoubleTensor asinInPlace();

    DoubleTensor acosInPlace();

    DoubleTensor expInPlace();

    DoubleTensor minusInPlace(DoubleTensor that);

    DoubleTensor plusInPlace(DoubleTensor that);

    DoubleTensor timesInPlace(DoubleTensor that);

    DoubleTensor divInPlace(DoubleTensor that);

    DoubleTensor unaryMinusInPlace();

    DoubleTensor absInPlace();

    DoubleTensor applyInPlace(Function<Double, Double> function);

    FlattenedView getFlattenedView();

    interface FlattenedView {

        long size();

        double get(long index);

        void set(long index, double value);

        double[] asArray();
    }
}
