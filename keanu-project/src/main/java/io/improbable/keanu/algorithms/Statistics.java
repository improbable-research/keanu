package io.improbable.keanu.algorithms;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class Statistics {

    private Map<Enum, List<Double>> statistics;

    public Statistics(Enum[] keys) {
        statistics = new HashMap<>();
        initialise(keys);
    }

    public void store(Enum key, Double value) {
        List<Double> values = statistics.get(key);
        values.add(value);
        statistics.put(key, values);
    }

    public List<Double> get(Enum key) {
        return statistics.get(key);
    }

    public Set<Enum> keys() {
        return statistics.keySet();
    }

    public double average(Enum key) {
        return statistics.get(key).stream().mapToDouble(x -> x).average().orElse(Double.NaN);
    }

    private void initialise(Enum[] keys) {
        for (Enum key : keys) {
            statistics.put(key, new ArrayList<>());
        }
    }
}
