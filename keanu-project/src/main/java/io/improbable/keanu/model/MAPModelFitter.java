package io.improbable.keanu.model;

import io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizer;

public class MAPModelFitter<INPUT, OUTPUT> implements ModelFitter<INPUT, OUTPUT> {
    @Override
    public void fit(ModelGraph<INPUT, OUTPUT> modelGraph, INPUT input, OUTPUT output) {
        modelGraph.observeValues(input, output);
        GradientOptimizer.of(modelGraph.getNet()).maxAPosteriori();
    }
}
