package io.improbable.keanu.algorithms.mcmc;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class Statistics {

    private Map<String, List<Double>> statistics;

    public Statistics(List<String> keys) {
        statistics = new HashMap<>();
        initialise(keys);
    }

    public void store(String key, Double value) {
        List<Double> values = statistics.get(key);
        values.add(value);
        statistics.put(key, values);
    }

    public List<Double> get(String key) {
        return statistics.get(key);
    }

    public Set<String> keys() {
        return statistics.keySet();
    }

    public double average(String key) {
        return statistics.get(key).stream().mapToDouble(x -> x).average().orElse(Double.NaN);
    }

    private Map<String, List<Double>> initialise(List<String> keys) {
        for (String key : keys) {
            statistics.put(key, new ArrayList<>());
        }
        return statistics;
    }
}
