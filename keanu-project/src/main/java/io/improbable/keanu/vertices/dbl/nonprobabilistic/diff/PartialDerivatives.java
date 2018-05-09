package io.improbable.keanu.vertices.dbl.nonprobabilistic.diff;

import io.improbable.keanu.vertices.Vertex;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

public class PartialDerivatives {

    public static PartialDerivatives OF_CONSTANT = new PartialDerivatives(Collections.emptyMap());

    public static PartialDerivatives withRespectToSelf(String withRespectTo) {
        return new PartialDerivatives(Collections.singletonMap(withRespectTo, 1.0));
    }

    private Map<String, Double> derivativeWithRespectTo;

    private PartialDerivatives(Map<String, Double> derivativeWithRespectTo) {
        this.derivativeWithRespectTo = derivativeWithRespectTo;
    }

    public Double withRespectTo(Vertex vertex) {
        return withRespectTo(vertex.getId());
    }

    public Double withRespectTo(String id) {
        return derivativeWithRespectTo.getOrDefault(id, 0.0);
    }

    public boolean isEmpty() {
        return derivativeWithRespectTo.isEmpty();
    }

    public Map<String, Double> asMap() {
        return derivativeWithRespectTo;
    }

    public void putWithRespectTo(String id, Double value) {
        derivativeWithRespectTo.put(id, value);
    }

    public PartialDerivatives add(PartialDerivatives toAdd) {
        Map<String, Double> added = copyPartialDerivatives(derivativeWithRespectTo);

        for (Map.Entry<String, Double> entry : toAdd.derivativeWithRespectTo.entrySet()) {
            String key = entry.getKey();
            added.put(key, added.getOrDefault(key, 0.0) + entry.getValue());
        }

        return new PartialDerivatives(added);
    }

    public PartialDerivatives subtract(PartialDerivatives toSubtract) {
        Map<String, Double> subtracted = copyPartialDerivatives(derivativeWithRespectTo);

        for (Map.Entry<String, Double> entry : toSubtract.derivativeWithRespectTo.entrySet()) {
            String key = entry.getKey();
            subtracted.put(key, subtracted.getOrDefault(key, 0.0) - entry.getValue());
        }

        return new PartialDerivatives(subtracted);
    }

    public PartialDerivatives multiplyBy(double multiplier) {
        Map<String, Double> multiplied = new HashMap<>();

        for (Map.Entry<String, Double> entry : derivativeWithRespectTo.entrySet()) {
            multiplied.put(
                    entry.getKey(),
                    entry.getValue() * multiplier
            );
        }

        return new PartialDerivatives(multiplied);
    }

    public PartialDerivatives divideBy(double divisor) {
        return multiplyBy(1.0 / divisor);
    }

    public PartialDerivatives powerTo(double power) {
        Map<String, Double> powered = new HashMap<>();

        for (Map.Entry<String, Double> entry : derivativeWithRespectTo.entrySet()) {
            powered.put(
                    entry.getKey(),
                    Math.pow(entry.getValue(), power)
            );
        }

        return new PartialDerivatives(powered);
    }

    public PartialDerivatives copy() {
        return new PartialDerivatives(copyPartialDerivatives(derivativeWithRespectTo));
    }

    private static Map<String, Double> copyPartialDerivatives(Map<String, Double> partialDerivatives) {
        return new HashMap<>(partialDerivatives);
    }
}
