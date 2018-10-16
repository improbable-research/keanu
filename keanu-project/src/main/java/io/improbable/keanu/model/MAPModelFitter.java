package io.improbable.keanu.model;

import io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizer;

public class MAPModelFitter<INPUT, OUTPUT> implements ModelFitter<INPUT, OUTPUT> {

    private final ModelGraph<INPUT, OUTPUT> modelGraph;

    public MAPModelFitter(ModelGraph<INPUT, OUTPUT> modelGraph) {
        this.modelGraph = modelGraph;
    }

    @Override
    public void fit(INPUT input, OUTPUT output) {
        modelGraph.observeValues(input, output);
        GradientOptimizer.of(modelGraph.getNet()).maxAPosteriori();
    }
}
