package io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff;

import io.improbable.keanu.vertices.dbltensor.DoubleTensor;

import java.util.HashMap;
import java.util.Map;

public class PartialDerivatives {

    private Map<String, DoubleTensor> partialDerivatives;

    public PartialDerivatives(String label, DoubleTensor infinitesimal) {
        this.partialDerivatives = new HashMap<>();
        this.partialDerivatives.put(label, infinitesimal);
    }

    public PartialDerivatives(Map<String, DoubleTensor> partialDerivatives) {
        this.partialDerivatives = partialDerivatives;
    }

    public Map<String, DoubleTensor> getPartialDerivatives() {
        return partialDerivatives;
    }

    public PartialDerivatives add(PartialDerivatives toAdd) {
        Map<String, DoubleTensor> added = cloneInfinitesimals(partialDerivatives);

        for (Map.Entry<String, DoubleTensor> entry : toAdd.partialDerivatives.entrySet()) {
            String k = entry.getKey();
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
        Map<String, DoubleTensor> subtracted = cloneInfinitesimals(partialDerivatives);

        for (Map.Entry<String, DoubleTensor> entry : toSubtract.partialDerivatives.entrySet()) {
            String k = entry.getKey();
            DoubleTensor v = entry.getValue();

            if (subtracted.containsKey(k)) {
                subtracted.put(k, subtracted.get(k).minus(v));
            } else {
                subtracted.put(k, v);
            }
        }

        return new PartialDerivatives(subtracted);
    }

    public PartialDerivatives multiplyBy(DoubleTensor multiplier) {
        Map<String, DoubleTensor> multiplied = new HashMap<>();

        for (Map.Entry<String, DoubleTensor> entry : partialDerivatives.entrySet()) {
            String k = entry.getKey();
            DoubleTensor v;
            if (entry.getValue().isScalar() && !multiplier.isScalar()) {
                v = entry.getValue().times(multiplier.sum());
            } else {
                v = entry.getValue().times(multiplier);
            }
            multiplied.put(k, v);
        }

        return new PartialDerivatives(multiplied);
    }

    public PartialDerivatives multiplyBy(double multiplier) {
        Map<String, DoubleTensor> multiplied = new HashMap<>();

        for (Map.Entry<String, DoubleTensor> entry : partialDerivatives.entrySet()) {
            String k = entry.getKey();
            DoubleTensor v = entry.getValue().times(multiplier);
            multiplied.put(k, v);
        }

        return new PartialDerivatives(multiplied);
    }

    public PartialDerivatives divideBy(DoubleTensor divisor) {
        Map<String, DoubleTensor> divided = new HashMap<>();

        for (Map.Entry<String, DoubleTensor> entry : partialDerivatives.entrySet()) {
            String k = entry.getKey();
            DoubleTensor v = entry.getValue().div(divisor);
            divided.put(k, v);
        }

        return new PartialDerivatives(divided);
    }

    public PartialDerivatives divideBy(double divisor) {
        Map<String, DoubleTensor> divided = new HashMap<>();

        for (Map.Entry<String, DoubleTensor> entry : partialDerivatives.entrySet()) {
            String k = entry.getKey();
            DoubleTensor v = entry.getValue().div(divisor);
            divided.put(k, v);
        }

        return new PartialDerivatives(divided);
    }

    public PartialDerivatives powerTo(double power) {
        Map<String, DoubleTensor> powered = new HashMap<>();

        for (Map.Entry<String, DoubleTensor> entry : partialDerivatives.entrySet()) {
            String k = entry.getKey();
            DoubleTensor v = entry.getValue().pow(power);
            powered.put(k, v);
        }

        return new PartialDerivatives(powered);
    }

    public PartialDerivatives clone() {
        return new PartialDerivatives(cloneInfinitesimals(partialDerivatives));
    }

    private static Map<String, DoubleTensor> cloneInfinitesimals(Map<String, DoubleTensor> infinitesimals) {
        Map<String, DoubleTensor> clone = new HashMap<>();
        for (Map.Entry<String, DoubleTensor> entry : infinitesimals.entrySet()) {
            clone.put(entry.getKey(), entry.getValue());
        }
        return clone;
    }
}
