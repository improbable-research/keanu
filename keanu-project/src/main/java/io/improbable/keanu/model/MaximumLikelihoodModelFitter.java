package io.improbable.keanu.model;

import io.improbable.keanu.Keanu;

public class MaximumLikelihoodModelFitter implements ModelFitter {

    /**
     * Uses the maximum likelihood algorithm to fit the {@link io.improbable.keanu.model.ModelGraph ModelGraph} to the input and output data.
     * This will mutate the graph which can then be used to construct a graph-backed model like, for instance a
     * {@link io.improbable.keanu.model.regression.RegressionModel RegressionModel}.
     *
     * @see <a href=https://en.wikipedia.org/wiki/Maximum_likelihood_estimation>Maximum Likelihood estimation</a>
     */
    @Override
    public void fit(ModelGraph modelGraph) {
        Keanu.Optimizer.Gradient.of(modelGraph.getBayesianNetwork()).maxLikelihood();
    }

}
