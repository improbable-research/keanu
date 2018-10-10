package io.improbable.keanu.model.regression;


import io.improbable.keanu.model.MAPModelFitter;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import lombok.experimental.UtilityClass;

/**
 * Utility for building a linear regression model.
 * <p>
 * This builds <a href="https://en.wikipedia.org/wiki/Tikhonov_regularization">ridge</a> linear regression, meaning that
 * the weights have a Gaussian prior distribution, i.e. the model has <a href="http://mathworld.wolfram.com/L2-Norm.html">L2 norm regularisation</a>.
 *
 * @see LinearRegressionModel
 */
@UtilityClass
public class LinearRidgeRegression {
    private static final double DEFAULT_MU = 0.0;
    private static final double DEFAULT_SIGMA = 2.0;

    public static LinearRidgeRegressionBuilder withFeatureShape(long... shape) {
        return new LinearRidgeRegressionBuilder(shape);
    }

    public static class LinearRidgeRegressionBuilder {
        final long[] shape;
        double[] priorOnWeightsSigmas;
        double[] priorOnWeightsMeans;
        double priorOnInterceptMean;
        double priorOnInterceptSigma;
        double observationSigma = DEFAULT_SIGMA;

        LinearRidgeRegressionBuilder(long[] shape) {
            this.shape = shape;
            setPriorOnWeightsAndIntercept(DEFAULT_MU, DEFAULT_SIGMA);
        }

        public LinearRidgeRegressionBuilder setPriorOnWeights(double[] means, double[] sigmas) {
            RegressionWeights.checkGaussianParameters(getFeatureCount(), means, sigmas);

            this.priorOnWeightsMeans = means;
            this.priorOnWeightsSigmas = sigmas;

            return this;
        }

        public LinearRidgeRegressionBuilder setPriorOnIntercept(double mean, double sigma) {
            this.priorOnInterceptMean = mean;
            this.priorOnInterceptSigma = sigma;
            return this;
        }

        public LinearRidgeRegressionBuilder setPriorOnWeightsAndIntercept(double mean, double sigma) {
            setPriorOnWeights(RegressionWeights.fillPriorOnWeights(this.shape, mean), RegressionWeights.fillPriorOnWeights(this.shape, sigma));
            setPriorOnIntercept(mean, sigma);
            return this;
        }

        public LinearRidgeRegressionBuilder setObservationSigma(double sigma) {
            this.observationSigma = sigma;
            return this;
        }

        public LinearRegressionModel build() {
            DoubleVertex interceptVertex = new GaussianVertex(priorOnInterceptMean, priorOnInterceptSigma);
            DoubleVertex weightsVertex = new GaussianVertex(new long[]{1, getFeatureCount()}, ConstantVertex.of(priorOnWeightsMeans), ConstantVertex.of(priorOnWeightsSigmas));
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
