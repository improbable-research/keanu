package io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff;

import io.improbable.keanu.vertices.dbltensor.DoubleTensor;

import java.util.HashMap;
import java.util.Map;

public class Infinitesimal {

    private Map<String, DoubleTensor> infinitesimals;

    public Infinitesimal(String label, DoubleTensor infinitesimal) {
        this.infinitesimals = new HashMap<>();
        this.infinitesimals.put(label, infinitesimal);
    }

    public Infinitesimal(Map<String, DoubleTensor> infinitesimals) {
        this.infinitesimals = infinitesimals;
    }

    public Map<String, DoubleTensor> getInfinitesimals() {
        return infinitesimals;
    }

    public Infinitesimal add(Infinitesimal toAdd) {
        Map<String, DoubleTensor> added = cloneInfinitesimals(infinitesimals);

        for (Map.Entry<String, DoubleTensor> entry : toAdd.infinitesimals.entrySet()) {
            String k = entry.getKey();
            DoubleTensor v = entry.getValue();
            added.put(k, added.getOrDefault(k, 0.0) + v);
        }

        return new Infinitesimal(added);
    }

    public Infinitesimal subtract(Infinitesimal toSubtract) {
        Map<String, DoubleTensor> subtracted = cloneInfinitesimals(infinitesimals);

        for (Map.Entry<String, DoubleTensor> entry : toSubtract.infinitesimals.entrySet()) {
            String k = entry.getKey();
            DoubleTensor v = entry.getValue();
            subtracted.put(k, subtracted.getOrDefault(k, 0.0) - v);
        }

        return new Infinitesimal(subtracted);
    }

    public Infinitesimal multiplyBy(double multiplier) {
        Map<String, DoubleTensor> multiplied = new HashMap<>();

        for (Map.Entry<String, DoubleTensor> entry : infinitesimals.entrySet()) {
            String k = entry.getKey();
            DoubleTensor v = entry.getValue() * multiplier;
            multiplied.put(k, v);
        }

        return new Infinitesimal(multiplied);
    }

    public Infinitesimal divideBy(double divisor) {
        return multiplyBy(1.0 / divisor);
    }

    public Infinitesimal powerTo(double power) {
        Map<String, DoubleTensor> powered = new HashMap<>();

        for (Map.Entry<String, DoubleTensor> entry : infinitesimals.entrySet()) {
            String k = entry.getKey();
            DoubleTensor v = Math.pow(entry.getValue(), power);
            powered.put(k, v);
        }

        return new Infinitesimal(powered);
    }

    public Infinitesimal clone() {
        return new Infinitesimal(cloneInfinitesimals(infinitesimals));
    }

    private static Map<String, DoubleTensor> cloneInfinitesimals(Map<String, DoubleTensor> infinitesimals) {
        Map<String, DoubleTensor> clone = new HashMap<>();
        for (Map.Entry<String, DoubleTensor> entry : infinitesimals.entrySet()) {
            clone.put(entry.getKey(), entry.getValue());
        }
        return clone;
    }
}
