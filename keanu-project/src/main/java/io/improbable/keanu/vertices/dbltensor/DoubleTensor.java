package io.improbable.keanu.vertices.dbltensor;

import java.util.HashMap;
import java.util.Map;

public interface DoubleTensor extends Tensor {

    static DoubleTensor create(double[] values, int[] shape) {
        return new Nd4jDoubleTensor(values, shape);
    }

    static DoubleTensor ones(int[] shape) {
        return Nd4jDoubleTensor.ones(shape);
    }

    static DoubleTensor zeros(int[] shape) {
        return Nd4jDoubleTensor.zeros(shape);
    }

    static DoubleTensor scalar(double scalarValue) {
        return new SimpleScalarTensor(scalarValue);
    }

    static DoubleTensor placeHolder(int[] shape) {
        return new Nd4jDoubleTensor(shape);
    }

    static Map<String, DoubleTensor> fromScalars(Map<String, Double> scalars) {
        Map<String, DoubleTensor> asTensors = new HashMap<>();

        for (Map.Entry<String, Double> entry : scalars.entrySet()) {
            asTensors.put(entry.getKey(), DoubleTensor.scalar(entry.getValue()));
        }

        return asTensors;
    }

    static Map<String, Double> toScalars(Map<String, DoubleTensor> tensors) {
        Map<String, Double> asScalars = new HashMap<>();

        for (Map.Entry<String, DoubleTensor> entry : tensors.entrySet()) {
            asScalars.put(entry.getKey(), entry.getValue().scalar());
        }

        return asScalars;
    }

    double getValue(int... index);

    void setValue(double value, int... index);

    double scalar();

    double sum();

    //New tensor Ops and transforms

    DoubleTensor reciprocal();

    DoubleTensor minus(double value);

    DoubleTensor plus(double value);

    DoubleTensor times(double value);

    DoubleTensor div(double value);

    DoubleTensor pow(DoubleTensor exponent);

    DoubleTensor pow(double exponent);

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

    DoubleTensor getLessThanOrEqualToMask(DoubleTensor lessThanThis);

    DoubleTensor applyWhere(DoubleTensor withMask, double value);

    //In place Ops and Transforms. These mutate the source vertex (i.e. this).

    DoubleTensor reciprocalInPlace();

    DoubleTensor minusInPlace(double value);

    DoubleTensor plusInPlace(double value);

    DoubleTensor timesInPlace(double value);

    DoubleTensor divInPlace(double value);

    DoubleTensor powInPlace(DoubleTensor exponent);

    DoubleTensor powInPlace(double exponent);

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

    double[] getLinearView();

}
