package io.improbable.keanu.model.regression;

import io.improbable.keanu.model.MAPModelFitter;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import lombok.experimental.UtilityClass;

/**
 * Utility for building a logistic regression model.
 * <p>
 * This builds <a href="https://en.wikipedia.org/wiki/Tikhonov_regularization">ridge</a> logistic regression, meaning that
 * the weights have a Gaussian prior distribution, i.e. the model has <a href="http://mathworld.wolfram.com/L2-Norm.html">L2 norm regularisation</a>.
 *
 * @see LogisticRegressionModel
 */
@UtilityClass
public class LogisticRidgeRegression {
    private static final double DEFAULT_MU = 0.0;
    private static final double DEFAULT_SIGMA = 2.0;

    public static LogisticRidgeRegressionBuilder withFeatureShape(long... shape) {
        return new LogisticRidgeRegressionBuilder(shape);
    }

    public static class LogisticRidgeRegressionBuilder {
        final long[] shape;
        double[] priorOnWeightsSigmas;
        double[] priorOnWeightsMeans;
        double priorOnInterceptMean;
        double priorOnInterceptSigma;

        LogisticRidgeRegressionBuilder(long[] shape) {
            this.shape = shape;
            setPriorOnWeightsAndIntercept(DEFAULT_MU, DEFAULT_SIGMA);
        }

        public LogisticRidgeRegressionBuilder setPriorOnWeights(double[] means, double[] sigmas) {
            RegressionWeights.checkGaussianParameters(getFeatureCount(), means, sigmas);

            this.priorOnWeightsMeans = means;
            this.priorOnWeightsSigmas = sigmas;

            return this;
        }

        public LogisticRidgeRegressionBuilder setPriorOnIntercept(double mean, double sigma) {
            this.priorOnInterceptMean = mean;
            this.priorOnInterceptSigma = sigma;
            return this;
        }

        public LogisticRidgeRegressionBuilder setPriorOnWeightsAndIntercept(double mean, double sigma) {
            setPriorOnWeights(RegressionWeights.fillPriorOnWeights(this.shape, mean), RegressionWeights.fillPriorOnWeights(this.shape, sigma));
            setPriorOnIntercept(mean, sigma);
            return this;
        }

        public LogisticRegressionModel build() {
            DoubleVertex interceptVertex = new GaussianVertex(priorOnInterceptMean, priorOnInterceptSigma);
            DoubleVertex weightsVertex = new GaussianVertex(new long[]{1, getFeatureCount()}, ConstantVertex.of(priorOnWeightsMeans), ConstantVertex.of(priorOnWeightsSigmas));
            LinearRegressionGraph<BooleanTensor> regressionGraph = new LinearRegressionGraph<>(
                this.shape,
                LogisticRegressionModel.logisticOutputTransform(),
                interceptVertex,
                weightsVertex
            );

            return new LogisticRegressionModel(regressionGraph, new MAPModelFitter<>());
        }

        private long getFeatureCount() {
            return this.shape[0];
        }
    }
}
