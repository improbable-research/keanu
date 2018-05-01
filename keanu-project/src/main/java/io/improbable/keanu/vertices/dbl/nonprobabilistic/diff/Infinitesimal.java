package io.improbable.keanu.vertices.dbl.nonprobabilistic.diff;

import java.util.HashMap;
import java.util.Map;

public class Infinitesimal {

    private Map<String, Double> infinitesimals;

    public Infinitesimal(String label, Double infinitesimal) {
        this.infinitesimals = new HashMap<>();
        this.infinitesimals.put(label, infinitesimal);
    }

    public Infinitesimal(Map<String, Double> infinitesimals) {
        this.infinitesimals = infinitesimals;
    }

    public Map<String, Double> getInfinitesimals() {
        return infinitesimals;
    }

    public Infinitesimal add(Infinitesimal toAdd) {
        Map<String, Double> added = copyInfinitesimals(infinitesimals);

        for (Map.Entry<String, Double> entry : toAdd.infinitesimals.entrySet()) {
            String k = entry.getKey();
            double v = entry.getValue();
            added.put(k, added.getOrDefault(k, 0.0) + v);
        }

        return new Infinitesimal(added);
    }

    public Infinitesimal subtract(Infinitesimal toSubtract) {
        Map<String, Double> subtracted = copyInfinitesimals(infinitesimals);

        for (Map.Entry<String, Double> entry : toSubtract.infinitesimals.entrySet()) {
            String k = entry.getKey();
            double v = entry.getValue();
            subtracted.put(k, subtracted.getOrDefault(k, 0.0) - v);
        }

        return new Infinitesimal(subtracted);
    }

    public Infinitesimal multiplyBy(double multiplier) {
        Map<String, Double> multiplied = new HashMap<>();

        for (Map.Entry<String, Double> entry : infinitesimals.entrySet()) {
            String k = entry.getKey();
            double v = entry.getValue() * multiplier;
            multiplied.put(k, v);
        }

        return new Infinitesimal(multiplied);
    }

    public Infinitesimal divideBy(double divisor) {
        return multiplyBy(1.0 / divisor);
    }

    public Infinitesimal powerTo(double power) {
        Map<String, Double> powered = new HashMap<>();

        for (Map.Entry<String, Double> entry : infinitesimals.entrySet()) {
            String k = entry.getKey();
            double v = Math.pow(entry.getValue(), power);
            powered.put(k, v);
        }

        return new Infinitesimal(powered);
    }

    public Infinitesimal copy() {
        return new Infinitesimal(copyInfinitesimals(infinitesimals));
    }

    private static Map<String, Double> copyInfinitesimals(Map<String, Double> infinitesimals) {
        Map<String, Double> clone = new HashMap<>();
        for (Map.Entry<String, Double> entry : infinitesimals.entrySet()) {
            clone.put(entry.getKey(), entry.getValue());
        }
        return clone;
    }
}
