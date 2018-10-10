package io.improbable.keanu.model.regression;

import io.improbable.keanu.model.MaximumLikelihoodModelFitter;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import lombok.experimental.UtilityClass;

/**
 * Utility for building a logistic regression model without regularisation.
 *
 * @see LogisticRegressionModel
 */
@UtilityClass
public class LogisticRegression {
    private static final double DEFAULT_MU = 0.0;
    private static final double DEFAULT_SIGMA = 2.0;

    public static LogisticRegressionBuilder withFeatureShape(long[] shape) {
        return new LogisticRegressionBuilder(shape);
    }

    public static class LogisticRegressionBuilder {
        final long[] shape;

        LogisticRegressionBuilder(long[] shape) {
            this.shape = shape;
        }

        public LogisticRegressionModel build() {
            DoubleVertex interceptVertex = new GaussianVertex(DEFAULT_MU, DEFAULT_SIGMA);
            DoubleVertex weightsVertex = new GaussianVertex(new long[]{1, getFeatureCount()}, ConstantVertex.of(DEFAULT_MU), ConstantVertex.of(DEFAULT_SIGMA));
            LinearRegressionGraph<BooleanTensor> regressionGraph = new LinearRegressionGraph<>(
                this.shape,
                LogisticRegressionModel.logisticOutputTransform(),
                interceptVertex,
                weightsVertex
            );

            return new LogisticRegressionModel(regressionGraph, new MaximumLikelihoodModelFitter<>());
        }

        private long getFeatureCount() {
            return this.shape[0];
        }
    }
}
