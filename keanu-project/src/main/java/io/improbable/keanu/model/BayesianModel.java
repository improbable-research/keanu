package io.improbable.keanu.model;

import io.improbable.keanu.network.BayesianNetwork;

abstract public class BayesianModel<INPUT, OUTPUT> implements Model<INPUT, OUTPUT> {
    private final ModelGraph<INPUT, OUTPUT> modelGraph;

    protected BayesianModel(ModelGraph<INPUT, OUTPUT> modelGraph) {
        this.modelGraph = modelGraph;
    }

    @Override
    public BayesianNetwork getNet() {
        return modelGraph.getNet();
    }

    @Override
    public OUTPUT predict(INPUT input) {
        return modelGraph.predict(input);
    }
}
