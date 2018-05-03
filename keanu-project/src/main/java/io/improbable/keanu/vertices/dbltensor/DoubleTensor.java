package io.improbable.keanu.vertices.dbltensor;

import io.improbable.keanu.kotlin.DoubleOperators;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.Map;

public interface DoubleTensor extends Tensor, DoubleOperators<DoubleTensor> {

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

    double getValue(int[] index);

    void setValue(double value, int[] index);

    double sum();

    double scalar();

    DoubleTensor reciprocal();
}
