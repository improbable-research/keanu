package io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

public class TensorPartialDerivatives {

    public static TensorPartialDerivatives OF_CONSTANT = new TensorPartialDerivatives(Collections.emptyMap());

    public static TensorPartialDerivatives withRespectToSelf(long withRespectTo, int[] shape) {
        return new TensorPartialDerivatives(Collections.singletonMap(withRespectTo, DoubleTensor.ones(shape)));
    }

    private Map<Long, DoubleTensor> derivativeWithRespectTo;

    public TensorPartialDerivatives(long id, DoubleTensor derivativeWithRespectTo) {
        this.derivativeWithRespectTo = new HashMap<>();
        this.derivativeWithRespectTo.put(id, derivativeWithRespectTo);
    }

    public TensorPartialDerivatives(Map<Long, DoubleTensor> derivativeWithRespectTo) {
        this.derivativeWithRespectTo = derivativeWithRespectTo;
    }

    public DoubleTensor withRespectTo(Vertex vertex) {
        return withRespectTo(vertex.getId());
    }

    public DoubleTensor withRespectTo(long id) {
        return derivativeWithRespectTo.getOrDefault(id, DoubleTensor.ZERO_SCALAR);
    }

    public boolean isEmpty() {
        return derivativeWithRespectTo.isEmpty();
    }

    public Map<Long, DoubleTensor> asMap() {
        return derivativeWithRespectTo;
    }

    public void putWithRespectTo(long id, DoubleTensor value) {
        derivativeWithRespectTo.put(id, value);
    }

    public TensorPartialDerivatives add(TensorPartialDerivatives toAdd) {
        Map<Long, DoubleTensor> added = cloneInfinitesimals(derivativeWithRespectTo);

        for (Map.Entry<Long, DoubleTensor> entry : toAdd.derivativeWithRespectTo.entrySet()) {
            long k = entry.getKey();
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
        Map<Long, DoubleTensor> subtracted = cloneInfinitesimals(derivativeWithRespectTo);

        for (Map.Entry<Long, DoubleTensor> entry : toSubtract.derivativeWithRespectTo.entrySet()) {
            long k = entry.getKey();
            DoubleTensor v = entry.getValue();

            if (subtracted.containsKey(k)) {
                subtracted.put(k, subtracted.get(k).minus(v));
            } else {
                subtracted.put(k, v.unaryMinus());
            }
        }

        return new TensorPartialDerivatives(subtracted);
    }

    public TensorPartialDerivatives multiplyBy(DoubleTensor multiplier) {
        Map<Long, DoubleTensor> multiplied = new HashMap<>();

        for (Map.Entry<Long, DoubleTensor> entry : derivativeWithRespectTo.entrySet()) {
            long k = entry.getKey();
            DoubleTensor v = entry.getValue().times(multiplier);
            multiplied.put(k, v);
        }

        return new TensorPartialDerivatives(multiplied);
    }

    public TensorPartialDerivatives multiplyBy(double multiplier) {
        Map<Long, DoubleTensor> multiplied = new HashMap<>();

        for (Map.Entry<Long, DoubleTensor> entry : derivativeWithRespectTo.entrySet()) {
            long k = entry.getKey();
            DoubleTensor v = entry.getValue().times(multiplier);
            multiplied.put(k, v);
        }

        return new TensorPartialDerivatives(multiplied);
    }

    public TensorPartialDerivatives divideBy(DoubleTensor divisor) {
        Map<Long, DoubleTensor> divided = new HashMap<>();

        for (Map.Entry<Long, DoubleTensor> entry : derivativeWithRespectTo.entrySet()) {
            long k = entry.getKey();
            DoubleTensor v = entry.getValue().div(divisor);
            divided.put(k, v);
        }

        return new TensorPartialDerivatives(divided);
    }

    public TensorPartialDerivatives divideBy(double divisor) {
        Map<Long, DoubleTensor> divided = new HashMap<>();

        for (Map.Entry<Long, DoubleTensor> entry : derivativeWithRespectTo.entrySet()) {
            long k = entry.getKey();
            DoubleTensor v = entry.getValue().div(divisor);
            divided.put(k, v);
        }

        return new TensorPartialDerivatives(divided);
    }

    public TensorPartialDerivatives powerTo(double power) {
        Map<Long, DoubleTensor> powered = new HashMap<>();

        for (Map.Entry<Long, DoubleTensor> entry : derivativeWithRespectTo.entrySet()) {
            long k = entry.getKey();
            DoubleTensor v = entry.getValue().pow(power);
            powered.put(k, v);
        }

        return new TensorPartialDerivatives(powered);
    }

    public TensorPartialDerivatives clone() {
        return new TensorPartialDerivatives(cloneInfinitesimals(derivativeWithRespectTo));
    }

    private static Map<Long, DoubleTensor> cloneInfinitesimals(Map<Long, DoubleTensor> infinitesimals) {
        Map<Long, DoubleTensor> clone = new HashMap<>();
        for (Map.Entry<Long, DoubleTensor> entry : infinitesimals.entrySet()) {
            clone.put(entry.getKey(), entry.getValue());
        }
        return clone;
    }
}
