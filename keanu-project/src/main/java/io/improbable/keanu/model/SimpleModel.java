package io.improbable.keanu.model;

import io.improbable.keanu.network.BayesianNetwork;

abstract public class SimpleModel<INPUT, OUTPUT> implements Model<INPUT, OUTPUT> {
    private final INPUT inputTrainingData;
    private final OUTPUT outputTrainingData;
    private final ModelGraph<INPUT, OUTPUT> modelGraph;
    private final ModelFitter<INPUT, OUTPUT> modelFitter;

    protected SimpleModel(INPUT inputTrainingData, OUTPUT outputTrainingData, ModelGraph<INPUT, OUTPUT> modelGraph, ModelFitter<INPUT, OUTPUT> modelFitter) {
        this.inputTrainingData = inputTrainingData;
        this.outputTrainingData = outputTrainingData;
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

    public void fit() {
        modelFitter.fit(modelGraph, inputTrainingData, outputTrainingData);
    }
}
