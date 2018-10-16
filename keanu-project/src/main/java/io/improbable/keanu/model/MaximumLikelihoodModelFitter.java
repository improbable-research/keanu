package io.improbable.keanu.model;

import io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizer;

public class MaximumLikelihoodModelFitter<INPUT, OUTPUT> implements ModelFitter<INPUT, OUTPUT> {

    private final ModelGraph<INPUT, OUTPUT> modelGraph;

    public MaximumLikelihoodModelFitter(ModelGraph<INPUT, OUTPUT> modelGraph) {
        this.modelGraph = modelGraph;
    }

    @Override
    public void fit(INPUT input, OUTPUT output) {
        modelGraph.observeValues(input, output);
        GradientOptimizer.of(modelGraph.getNet()).maxLikelihood();
    }
}

