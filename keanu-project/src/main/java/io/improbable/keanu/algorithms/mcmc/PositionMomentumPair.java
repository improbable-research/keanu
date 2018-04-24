package io.improbable.keanu.algorithms.mcmc;

import java.util.Map;

public class PositionMomentumPair {

    private final Map<String, Double> position;
    private final Map<String, Double> momentum;

    public PositionMomentumPair(Map<String, Double> position, Map<String, Double> momentum) {
        this.position = position;
        this.momentum = momentum;
    }

    public Map<String, Double> getPosition() {
        return position;
    }

    public Map<String, Double> getMomentum() {
        return momentum;
    }
}
