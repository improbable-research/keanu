package io.improbable.keanu.model.regression;

import io.improbable.keanu.model.MAPModelFitter;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.LaplaceVertex;
import lombok.experimental.UtilityClass;

/**
 * Utility for building a logistic regression model.
 * <p>
 * This builds <a href="https://en.wikipedia.org/wiki/Lasso_(statistics)">lasso</a> logistic regression, meaning that
 * the weights have a Laplace prior distribution, i.e. the model has <a href="http://mathworld.wolfram.com/L1-Norm.html">L1 norm regularisation</a>.
 *
 * @see LogisticRegressionModel
 */
@UtilityClass
public class LogisticLassoRegression {
    private static final double DEFAULT_MU = 0.0;
    private static final double DEFAULT_BETA = 2.0;

    public static LogisticLassoRegressionBuilder withFeatureShape(long... shape) {
        return new LogisticLassoRegressionBuilder(shape);
    }

    public static class LogisticLassoRegressionBuilder {
        final long[] shape;
        double[] priorOnWeightsBetas;
        double[] priorOnWeightsMeans;
        double priorOnInterceptMean;
        double priorOnInterceptBeta;

        LogisticLassoRegressionBuilder(long[] shape) {
            this.shape = shape;
            setPriorOnWeightsAndIntercept(DEFAULT_MU, DEFAULT_BETA);
        }

        public LogisticLassoRegressionBuilder setPriorOnWeights(double[] means, double[] betas) {
            RegressionWeights.checkLaplaceParameters(getFeatureCount(), means, betas);

            this.priorOnWeightsMeans = means;
            this.priorOnWeightsBetas = betas;

            return this;
        }

        public LogisticLassoRegressionBuilder setPriorOnIntercept(double mean, double beta) {
            this.priorOnInterceptMean = mean;
            this.priorOnInterceptBeta = beta;
            return this;
        }

        public LogisticLassoRegressionBuilder setPriorOnWeightsAndIntercept(double mean, double beta) {
            setPriorOnWeights(RegressionWeights.fillPriorOnWeights(this.shape, mean), RegressionWeights.fillPriorOnWeights(this.shape, beta));
            setPriorOnIntercept(mean, beta);
            return this;
        }

        public LogisticRegressionModel build() {
            DoubleVertex interceptVertex = new LaplaceVertex(priorOnInterceptMean, priorOnInterceptBeta);
            DoubleVertex weightsVertex = new LaplaceVertex(new long[]{1, getFeatureCount()}, ConstantVertex.of(priorOnWeightsMeans), ConstantVertex.of(priorOnWeightsBetas));
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
