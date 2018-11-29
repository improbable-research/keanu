package io.improbable.keanu.model;

import io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizer;

public class MaximumLikelihoodModelFitter<INPUT, OUTPUT> implements ModelFitter<INPUT, OUTPUT> {

    private final ModelGraph<INPUT, OUTPUT> modelGraph;

    public MaximumLikelihoodModelFitter(ModelGraph<INPUT, OUTPUT> modelGraph) {
        this.modelGraph = modelGraph;
    }

    /**
     * Uses the maximum likelihood algorithm to fit the {@link io.improbable.keanu.model.ModelGraph ModelGraph} to the input and output data.
     * This will mutate the graph which can then be used to construct a graph-backed model like, for instance a
     * {@link io.improbable.keanu.model.regression.RegressionModel RegressionModel}.
     *
     * @see <a href=https://en.wikipedia.org/wiki/Maximum_likelihood_estimation>Maximum Likelihood estimation</a>
     */
    @Override
    public void fit() {
        GradientOptimizer.of(modelGraph.getBayesianNetwork()).maxLikelihood();
    }

}
