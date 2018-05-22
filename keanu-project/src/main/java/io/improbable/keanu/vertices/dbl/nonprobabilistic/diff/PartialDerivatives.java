package io.improbable.keanu.vertices.dbl.nonprobabilistic.diff;

import io.improbable.keanu.vertices.Vertex;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

public class PartialDerivatives {

    public static PartialDerivatives OF_CONSTANT = new PartialDerivatives(Collections.emptyMap());

    public static PartialDerivatives withRespectToSelf(long withRespectTo) {
        return new PartialDerivatives(Collections.singletonMap(withRespectTo, 1.0));
    }

    private Map<Long, Double> derivativeWithRespectTo;

    private PartialDerivatives(Map<Long, Double> derivativeWithRespectTo) {
        this.derivativeWithRespectTo = derivativeWithRespectTo;
    }

    public Double withRespectTo(Vertex vertex) {
        return withRespectTo(vertex.getId());
    }

    public Double withRespectTo(long id) {
        return derivativeWithRespectTo.getOrDefault(id, 0.0);
    }

    public boolean isEmpty() {
        return derivativeWithRespectTo.isEmpty();
    }

    public Map<Long, Double> asMap() {
        return derivativeWithRespectTo;
    }

    public void putWithRespectTo(long id, Double value) {
        derivativeWithRespectTo.put(id, value);
    }

    public PartialDerivatives add(PartialDerivatives toAdd) {
        Map<Long, Double> added = copyPartialDerivatives(derivativeWithRespectTo);

        for (Map.Entry<Long, Double> entry : toAdd.derivativeWithRespectTo.entrySet()) {
            long key = entry.getKey();
            added.put(key, added.getOrDefault(key, 0.0) + entry.getValue());
        }

        return new PartialDerivatives(added);
    }

    public PartialDerivatives subtract(PartialDerivatives toSubtract) {
        Map<Long, Double> subtracted = copyPartialDerivatives(derivativeWithRespectTo);

        for (Map.Entry<Long, Double> entry : toSubtract.derivativeWithRespectTo.entrySet()) {
            long key = entry.getKey();
            subtracted.put(key, subtracted.getOrDefault(key, 0.0) - entry.getValue());
        }

        return new PartialDerivatives(subtracted);
    }

    public PartialDerivatives multiplyBy(double multiplier) {
        Map<Long, Double> multiplied = new HashMap<>();

        for (Map.Entry<Long, Double> entry : derivativeWithRespectTo.entrySet()) {
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
        Map<Long, Double> powered = new HashMap<>();

        for (Map.Entry<Long, Double> entry : derivativeWithRespectTo.entrySet()) {
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

    private static Map<Long, Double> copyPartialDerivatives(Map<Long, Double> partialDerivatives) {
        return new HashMap<>(partialDerivatives);
    }
}
