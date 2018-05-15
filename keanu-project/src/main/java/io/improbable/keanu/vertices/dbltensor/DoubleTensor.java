package io.improbable.keanu.vertices.dbltensor;

import java.util.HashMap;
import java.util.Map;

public interface DoubleTensor extends Tensor {

    static DoubleTensor scalar(double scalarValue) {
        return new SimpleScalarTensor(scalarValue);
    }

    static Map<Long, DoubleTensor> fromScalars(Map<Long, Double> scalars) {
        Map<Long, DoubleTensor> asTensors = new HashMap<>();

        for (Map.Entry<Long, Double> entry : scalars.entrySet()) {
            asTensors.put(entry.getKey(), DoubleTensor.scalar(entry.getValue()));
        }

        return asTensors;
    }

    double scalar();

}
