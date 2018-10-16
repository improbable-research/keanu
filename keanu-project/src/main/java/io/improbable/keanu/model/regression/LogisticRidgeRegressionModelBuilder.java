package io.improbable.keanu.model.regression;

import io.improbable.keanu.model.MAPModelFitter;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

/**
 * Utility for building a logistic regression model.
 * <p>
 * This builds <a href="https://en.wikipedia.org/wiki/Tikhonov_regularization">ridge</a> logistic regression, meaning that
 * the weights have a Gaussian prior distribution, i.e. the model has <a href="http://mathworld.wolfram.com/L2-Norm.html">L2 norm regularisation</a>.
 *
 * @see LogisticRegressionModel
 */
public class LogisticRidgeRegressionModelBuilder {
    private static final double DEFAULT_MU = 0.0;
    private static final double DEFAULT_SIGMA = 2.0;

    private DoubleTensor inputTrainingData;
    private BooleanTensor outputTrainingData;


    double[] priorOnWeightsSigmas;
    double[] priorOnWeightsMeans;
    double priorOnInterceptMean;
    double priorOnInterceptSigma;

    public LogisticRidgeRegressionModelBuilder() {
        setPriorOnWeightsAndIntercept(DEFAULT_MU, DEFAULT_SIGMA);
    }

    public LogisticRidgeRegressionModelBuilder setInputTrainingData(DoubleTensor inputTrainingData) {
        this.inputTrainingData = inputTrainingData;
        return this;
    }

    public LogisticRidgeRegressionModelBuilder setOuputTrainingData(BooleanTensor outputTrainingData) {
        this.outputTrainingData = outputTrainingData;
        return this;
    }

    public LogisticRidgeRegressionModelBuilder setPriorOnWeights(double[] means, double[] sigmas) {
        RegressionWeights.checkGaussianParameters(getFeatureCount(), means, sigmas);

        this.priorOnWeightsMeans = means;
        this.priorOnWeightsSigmas = sigmas;

        return this;
    }

    public LogisticRidgeRegressionModelBuilder setPriorOnIntercept(double mean, double sigma) {
        this.priorOnInterceptMean = mean;
        this.priorOnInterceptSigma = sigma;
        return this;
    }

    public LogisticRidgeRegressionModelBuilder setPriorOnWeightsAndIntercept(double mean, double sigma) {
        setPriorOnWeights(RegressionWeights.fillPriorOnWeights(this.inputTrainingData.getShape(), mean), RegressionWeights.fillPriorOnWeights(this.inputTrainingData.getShape(), sigma));
        setPriorOnIntercept(mean, sigma);
        return this;
    }

    public LogisticRegressionModel build() {
        DoubleVertex interceptVertex = new GaussianVertex(priorOnInterceptMean, priorOnInterceptSigma);
        DoubleVertex weightsVertex = new GaussianVertex(new long[]{1, getFeatureCount()}, ConstantVertex.of(priorOnWeightsMeans), ConstantVertex.of(priorOnWeightsSigmas));
        LinearRegressionGraph<BooleanTensor> regressionGraph = new LinearRegressionGraph<>(
            this.inputTrainingData.getShape(),
            LogisticRegressionModel.logisticOutputTransform(),
            interceptVertex,
            weightsVertex
        );

        MAPModelFitter<DoubleTensor, BooleanTensor> fitter = new MAPModelFitter<>(regressionGraph);
        fitter.fit(inputTrainingData, outputTrainingData);
        return new LogisticRegressionModel(regressionGraph);
    }

    private long getFeatureCount() {
            return this.inputTrainingData.getShape()[0];
        }
}
