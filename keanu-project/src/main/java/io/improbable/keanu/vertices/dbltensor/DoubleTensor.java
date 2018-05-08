package io.improbable.keanu.vertices.dbltensor;

import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.Map;

public interface DoubleTensor extends Tensor {

    static DoubleTensor create(double[] values, int[] shape) {
        return new Nd4jDoubleTensor(values, shape);
    }

    static DoubleTensor ones(int[] shape) {
        return new Nd4jDoubleTensor(Nd4j.ones(shape));
    }

    static DoubleTensor zeros(int[] shape) {
        return new Nd4jDoubleTensor(Nd4j.zeros(shape));
    }

    static DoubleTensor scalar(double scalarValue) {
        return new SimpleScalarTensor(scalarValue);
    }

    static DoubleTensor placeHolder(int[] shape) {
        return new Nd4jDoubleTensor(shape);
    }

    static DoubleTensor nd4JScalar(double scalarValue) {
        return new Nd4jDoubleTensor(Nd4j.scalar(scalarValue));
    }

    static Map<String, DoubleTensor> fromScalars(Map<String, Double> scalars) {
        Map<String, DoubleTensor> asTensors = new HashMap<>();

        for (Map.Entry<String, Double> entry : scalars.entrySet()) {
            asTensors.put(entry.getKey(), DoubleTensor.scalar(entry.getValue()));
        }

        return asTensors;
    }

    static Map<String, Double> toScalars(Map<String, DoubleTensor> tensors) {
        Map<String, Double> asTensors = new HashMap<>();

        for (Map.Entry<String, DoubleTensor> entry : tensors.entrySet()) {
            asTensors.put(entry.getKey(), entry.getValue().scalar());
        }

        return asTensors;
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

    //In place Ops and Transforms
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
