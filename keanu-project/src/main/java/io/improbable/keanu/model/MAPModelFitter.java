package io.improbable.keanu.model;

import io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizer;

public class MAPModelFitter implements ModelFitter {

    /**
     * Uses the Maximum A Posteriori algorithm to fit the model graph to the input and output data.
     * This will mutate the graph which can then be used to construct a graph-backed model like, for instance, a
     * {@link io.improbable.keanu.model.regression.RegressionModel RegressionModel}
     *
     * @see <a href=https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation>Maximum A Posteriori estimation</a>
     */
    @Override
    public void fit(ModelGraph modelGraph) {
        GradientOptimizer.of(modelGraph.getBayesianNetwork()).maxAPosteriori();
    }
}
