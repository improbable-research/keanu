package io.improbable.keanu.model;

import io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizer;

public class MaximumLikelihoodModelFitter<INPUT, OUTPUT> implements ModelFitter<INPUT, OUTPUT> {

    private final ModelGraph<INPUT, OUTPUT> modelGraph;

    public MaximumLikelihoodModelFitter(ModelGraph<INPUT, OUTPUT> modelGraph) {
        this.modelGraph = modelGraph;
    }

    /**
     * Uses the maximum likelihood algorithm to fit the model graph to a given set of input and output data.
     * This will mutate the graph which can then be used to construct
     * {@link io.improbable.keanu.model.regression.LogisticRegressionModel LogisticRegressionModel} or
     * {@link io.improbable.keanu.model.regression.LinearRegressionModel LinearRegressionModel} objects
     *
     * @param input The input data to your model graph
     * @param output The output data to your model graph
     *
     * @see <a href=https://en.wikipedia.org/wiki/Maximum_likelihood_estimation>Maximum Likelihood estimation</a>
     */
    @Override
    public void fit(INPUT input, OUTPUT output) {
        modelGraph.observeValues(input, output);
        GradientOptimizer.of(modelGraph.getNet()).maxLikelihood();
    }
}

