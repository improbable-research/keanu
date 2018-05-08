package io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.Nd4jDoubleTensor;

import java.util.HashMap;
import java.util.Map;

public class TensorPartialDerivatives {

    private Map<String, DoubleTensor> derivativeWithRespectTo;

    public TensorPartialDerivatives(String label, DoubleTensor derivativeWithRespectTo) {
        this.derivativeWithRespectTo = new HashMap<>();
        this.derivativeWithRespectTo.put(label, derivativeWithRespectTo);
    }

    public TensorPartialDerivatives(Map<String, DoubleTensor> derivativeWithRespectTo) {
        this.derivativeWithRespectTo = derivativeWithRespectTo;
    }

    public DoubleTensor withRespectTo(Vertex vertex) {
        return withRespectTo(vertex.getId());
    }

    public DoubleTensor withRespectTo(String id) {
        return derivativeWithRespectTo.getOrDefault(id, Nd4jDoubleTensor.ZERO_SCALAR);
    }

    public boolean isEmpty() {
        return derivativeWithRespectTo.isEmpty();
    }

    public Map<String, DoubleTensor> asMap() {
        return derivativeWithRespectTo;
    }

    public void putWithRespectTo(String id, DoubleTensor value) {
        derivativeWithRespectTo.put(id, value);
    }

    public TensorPartialDerivatives add(TensorPartialDerivatives toAdd) {
        Map<String, DoubleTensor> added = cloneInfinitesimals(derivativeWithRespectTo);

        for (Map.Entry<String, DoubleTensor> entry : toAdd.derivativeWithRespectTo.entrySet()) {
            String k = entry.getKey();
            DoubleTensor v = entry.getValue();

            if (added.containsKey(k)) {
                added.put(k, added.get(k).plus(v));
            } else {
                added.put(k, v);
            }
        }

        return new TensorPartialDerivatives(added);
    }

    public TensorPartialDerivatives subtract(TensorPartialDerivatives toSubtract) {
        Map<String, DoubleTensor> subtracted = cloneInfinitesimals(derivativeWithRespectTo);

        for (Map.Entry<String, DoubleTensor> entry : toSubtract.derivativeWithRespectTo.entrySet()) {
            String k = entry.getKey();
            DoubleTensor v = entry.getValue();

            if (subtracted.containsKey(k)) {
                subtracted.put(k, subtracted.get(k).minus(v));
            } else {
                subtracted.put(k, v);
            }
        }

        return new TensorPartialDerivatives(subtracted);
    }

    public TensorPartialDerivatives multiplyBy(DoubleTensor multiplier) {
        Map<String, DoubleTensor> multiplied = new HashMap<>();

        for (Map.Entry<String, DoubleTensor> entry : derivativeWithRespectTo.entrySet()) {
            String k = entry.getKey();
            DoubleTensor v;
            if (entry.getValue().isScalar() && !multiplier.isScalar()) {
                v = entry.getValue().times(multiplier.sum());
            } else {
                v = entry.getValue().times(multiplier);
            }
            multiplied.put(k, v);
        }

        return new TensorPartialDerivatives(multiplied);
    }

    public TensorPartialDerivatives multiplyBy(double multiplier) {
        Map<String, DoubleTensor> multiplied = new HashMap<>();

        for (Map.Entry<String, DoubleTensor> entry : derivativeWithRespectTo.entrySet()) {
            String k = entry.getKey();
            DoubleTensor v = entry.getValue().times(multiplier);
            multiplied.put(k, v);
        }

        return new TensorPartialDerivatives(multiplied);
    }

    public TensorPartialDerivatives divideBy(DoubleTensor divisor) {
        Map<String, DoubleTensor> divided = new HashMap<>();

        for (Map.Entry<String, DoubleTensor> entry : derivativeWithRespectTo.entrySet()) {
            String k = entry.getKey();
            DoubleTensor v = entry.getValue().div(divisor);
            divided.put(k, v);
        }

        return new TensorPartialDerivatives(divided);
    }

    public TensorPartialDerivatives divideBy(double divisor) {
        Map<String, DoubleTensor> divided = new HashMap<>();

        for (Map.Entry<String, DoubleTensor> entry : derivativeWithRespectTo.entrySet()) {
            String k = entry.getKey();
            DoubleTensor v = entry.getValue().div(divisor);
            divided.put(k, v);
        }

        return new TensorPartialDerivatives(divided);
    }

    public TensorPartialDerivatives powerTo(double power) {
        Map<String, DoubleTensor> powered = new HashMap<>();

        for (Map.Entry<String, DoubleTensor> entry : derivativeWithRespectTo.entrySet()) {
            String k = entry.getKey();
            DoubleTensor v = entry.getValue().pow(power);
            powered.put(k, v);
        }

        return new TensorPartialDerivatives(powered);
    }

    public TensorPartialDerivatives clone() {
        return new TensorPartialDerivatives(cloneInfinitesimals(derivativeWithRespectTo));
    }

    private static Map<String, DoubleTensor> cloneInfinitesimals(Map<String, DoubleTensor> infinitesimals) {
        Map<String, DoubleTensor> clone = new HashMap<>();
        for (Map.Entry<String, DoubleTensor> entry : infinitesimals.entrySet()) {
            clone.put(entry.getKey(), entry.getValue());
        }
        return clone;
    }
}
