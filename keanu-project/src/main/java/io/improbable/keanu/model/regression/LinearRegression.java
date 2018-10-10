package io.improbable.keanu.model.regression;

import io.improbable.keanu.model.MaximumLikelihoodModelFitter;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import lombok.experimental.UtilityClass;

/**
 * Utility for building a linear regression model without regularisation.
 *
 * @see LinearRegressionModel
 */
@UtilityClass
public class LinearRegression {
    private static final double DEFAULT_MU = 0.0;
    private static final double DEFAULT_SIGMA = 2.0;

    public static LinearRegressionBuilder withFeatureShape(long[] shape) {
        return new LinearRegressionBuilder(shape);
    }

    public static class LinearRegressionBuilder {
        final long[] shape;
        double observationSigma = DEFAULT_SIGMA;

        LinearRegressionBuilder(long[] shape) {
            this.shape = shape;
        }

        public LinearRegressionBuilder setObservationSigma(double sigma) {
            this.observationSigma = sigma;
            return this;
        }

        public LinearRegressionModel build() {
            DoubleVertex interceptVertex = new GaussianVertex(DEFAULT_MU, DEFAULT_SIGMA);
            DoubleVertex weightsVertex = new GaussianVertex(new long[]{1, getFeatureCount()}, ConstantVertex.of(DEFAULT_MU), ConstantVertex.of(DEFAULT_SIGMA));
            LinearRegressionGraph<DoubleTensor> regressionGraph = new LinearRegressionGraph<>(
                this.shape,
                LinearRegressionModel.gaussianOutputTransform(observationSigma),
                interceptVertex,
                weightsVertex
            );

            return new LinearRegressionModel(regressionGraph, new MaximumLikelihoodModelFitter<>());
        }

        private long getFeatureCount() {
            return this.shape[0];
        }
    }
}
