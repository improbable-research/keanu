package io.improbable.keanu.vertices.dbl.nonprobabilistic.diff;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

public class PartialDerivatives {

    public static PartialDerivatives OF_CONSTANT = new PartialDerivatives(Collections.emptyMap());

    public static PartialDerivatives withRespectToSelf(long withRespectTo, int[] shape) {
        return new PartialDerivatives(Collections.singletonMap(withRespectTo, DoubleTensor.ones(shape)));
    }

    private Map<Long, DoubleTensor> derivativeWithRespectTo;

    public PartialDerivatives(long id, DoubleTensor derivativeWithRespectTo) {
        this.derivativeWithRespectTo = new HashMap<>();
        this.derivativeWithRespectTo.put(id, derivativeWithRespectTo);
    }

    public PartialDerivatives(Map<Long, DoubleTensor> derivativeWithRespectTo) {
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

    public PartialDerivatives add(PartialDerivatives toAdd) {
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

        return new PartialDerivatives(added);
    }

    public PartialDerivatives subtract(PartialDerivatives toSubtract) {
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

        return new PartialDerivatives(subtracted);
    }

    public PartialDerivatives multiplyBy(DoubleTensor multiplier) {
        Map<Long, DoubleTensor> multiplied = new HashMap<>();

        for (Map.Entry<Long, DoubleTensor> entry : derivativeWithRespectTo.entrySet()) {
            long k = entry.getKey();
            DoubleTensor v = entry.getValue().times(multiplier);
            multiplied.put(k, v);
        }

        return new PartialDerivatives(multiplied);
    }

    public PartialDerivatives multiplyBy(double multiplier) {
        Map<Long, DoubleTensor> multiplied = new HashMap<>();

        for (Map.Entry<Long, DoubleTensor> entry : derivativeWithRespectTo.entrySet()) {
            long k = entry.getKey();
            DoubleTensor v = entry.getValue().times(multiplier);
            multiplied.put(k, v);
        }

        return new PartialDerivatives(multiplied);
    }

    public PartialDerivatives divideBy(DoubleTensor divisor) {
        Map<Long, DoubleTensor> divided = new HashMap<>();

        for (Map.Entry<Long, DoubleTensor> entry : derivativeWithRespectTo.entrySet()) {
            long k = entry.getKey();
            DoubleTensor v = entry.getValue().div(divisor);
            divided.put(k, v);
        }

        return new PartialDerivatives(divided);
    }

    public PartialDerivatives divideBy(double divisor) {
        Map<Long, DoubleTensor> divided = new HashMap<>();

        for (Map.Entry<Long, DoubleTensor> entry : derivativeWithRespectTo.entrySet()) {
            long k = entry.getKey();
            DoubleTensor v = entry.getValue().div(divisor);
            divided.put(k, v);
        }

        return new PartialDerivatives(divided);
    }

    public PartialDerivatives powerTo(double power) {
        Map<Long, DoubleTensor> powered = new HashMap<>();

        for (Map.Entry<Long, DoubleTensor> entry : derivativeWithRespectTo.entrySet()) {
            long k = entry.getKey();
            DoubleTensor v = entry.getValue().pow(power);
            powered.put(k, v);
        }

        return new PartialDerivatives(powered);
    }

    public PartialDerivatives clone() {
        return new PartialDerivatives(cloneInfinitesimals(derivativeWithRespectTo));
    }

    private static Map<Long, DoubleTensor> cloneInfinitesimals(Map<Long, DoubleTensor> infinitesimals) {
        Map<Long, DoubleTensor> clone = new HashMap<>();
        for (Map.Entry<Long, DoubleTensor> entry : infinitesimals.entrySet()) {
            clone.put(entry.getKey(), entry.getValue());
        }
        return clone;
    }
}
