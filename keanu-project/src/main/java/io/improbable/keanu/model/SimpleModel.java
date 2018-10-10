package io.improbable.keanu.model;

import io.improbable.keanu.network.BayesianNetwork;

abstract public class SimpleModel<INPUT, OUTPUT> implements Model<INPUT, OUTPUT> {
    private final ModelGraph<INPUT, OUTPUT> modelGraph;
    private final ModelFitter<INPUT, OUTPUT> modelFitter;

    protected SimpleModel(ModelGraph<INPUT, OUTPUT> modelGraph, ModelFitter<INPUT, OUTPUT> modelFitter) {
        this.modelGraph = modelGraph;
        this.modelFitter = modelFitter;
    }

    @Override
    public BayesianNetwork getNet() {
        return modelGraph.getNet();
    }

    @Override
    public OUTPUT predict(INPUT input) {
        return modelGraph.predict(input);
    }

    public void fit(INPUT input, OUTPUT output) {
        modelFitter.fit(modelGraph, input, output);
    }
}
