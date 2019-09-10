package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import lombok.Value;

@Value
public class FitnessAndGradientFlat {

    double fitness;
    double[] gradient;
}
