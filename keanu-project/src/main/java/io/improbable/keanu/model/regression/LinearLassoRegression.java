package io.improbable.keanu.model.regression;

import io.improbable.keanu.model.MAPModelFitter;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.LaplaceVertex;
import lombok.experimental.UtilityClass;

/**
 * Utility for building a linear regression model.
 * <p>
 * This builds <a href="https://en.wikipedia.org/wiki/Lasso_(statistics)">lasso</a> linear regression, meaning that
 * the weights have a Laplace prior distribution, i.e. the model has <a href="http://mathworld.wolfram.com/L1-Norm.html">L1 norm regularisation</a>.
 *
 * @see LinearRegressionModel
 */
@UtilityClass
public class LinearLassoRegression {
    private static final double DEFAULT_MU = 0.0;
    private static final double DEFAULT_BETA = 2.0;

    public static LinearLassoRegressionBuilder withFeatureShape(long... shape) {
        return new LinearLassoRegressionBuilder(shape);
    }

    public static class LinearLassoRegressionBuilder {
        final long[] shape;
        double[] priorOnWeightsBetas;
        double[] priorOnWeightsMeans;
        double priorOnInterceptMean;
        double priorOnInterceptBeta;
        double observationSigma = DEFAULT_BETA;

        LinearLassoRegressionBuilder(long[] shape) {
            this.shape = shape;
            setPriorOnWeightsAndIntercept(DEFAULT_MU, DEFAULT_BETA);
        }

        public LinearLassoRegressionBuilder setPriorOnWeights(double[] means, double[] betas) {
            RegressionWeights.checkLaplaceParameters(getFeatureCount(), means, betas);

            this.priorOnWeightsMeans = means;
            this.priorOnWeightsBetas = betas;

            return this;
        }

        public LinearLassoRegressionBuilder setPriorOnIntercept(double mean, double beta) {
            this.priorOnInterceptMean = mean;
            this.priorOnInterceptBeta = beta;
            return this;
        }

        public LinearLassoRegressionBuilder setPriorOnWeightsAndIntercept(double mean, double beta) {
            setPriorOnWeights(RegressionWeights.fillPriorOnWeights(this.shape, mean), RegressionWeights.fillPriorOnWeights(this.shape, beta));
            setPriorOnIntercept(mean, beta);
            return this;
        }

        public LinearLassoRegressionBuilder setObservationSigma(double sigma) {
            this.observationSigma = sigma;
            return this;
        }

        public LinearRegressionModel build() {
            DoubleVertex interceptVertex = new LaplaceVertex(priorOnInterceptMean, priorOnInterceptBeta);
            DoubleVertex weightsVertex = new LaplaceVertex(new long[]{1, getFeatureCount()}, ConstantVertex.of(priorOnWeightsMeans), ConstantVertex.of(priorOnWeightsBetas));
            LinearRegressionGraph<DoubleTensor> regressionGraph = new LinearRegressionGraph<>(
                this.shape,
                LinearRegressionModel.gaussianOutputTransform(observationSigma),
                interceptVertex,
                weightsVertex
            );
            return new LinearRegressionModel(regressionGraph, new MAPModelFitter<>());
        }

        private long getFeatureCount() {
            return this.shape[0];
        }
    }
}
