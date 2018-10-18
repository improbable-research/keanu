package io.improbable.keanu.model.regression;


import io.improbable.keanu.model.MAPModelFitter;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

/**
 * Builder class for creating a linear ridge regression model.
 * <p>
 * This builds <a href="https://en.wikipedia.org/wiki/Tikhonov_regularization">ridge</a> linear regression, meaning that
 * the weights have a Gaussian prior distribution, i.e. the model has <a href="http://mathworld.wolfram.com/L2-Norm.html">L2 norm regularisation</a>.
 *
 * @see LinearRegressionModel
 */
public class LinearRidgeRegressionModelBuilder {
    private static final double DEFAULT_MU = 0.0;
    private static final double DEFAULT_SIGMA = 2.0;

    private DoubleTensor inputTrainingData;
    private DoubleTensor outputTrainingData;

    double[] priorOnWeightsSigmas;
    double[] priorOnWeightsMeans;
    Double priorOnInterceptMean;
    Double priorOnInterceptSigma;
    double observationSigma = DEFAULT_SIGMA;


    public LinearRidgeRegressionModelBuilder setInputTrainingData(DoubleTensor inputTrainingData) {
        this.inputTrainingData = inputTrainingData;
        return this;
    }

    public LinearRidgeRegressionModelBuilder setOutputTrainingData(DoubleTensor outputTrainingData) {
        this.outputTrainingData = outputTrainingData;
        return this;
    }

    /**
     * Set the input parameters to the gaussian distribution describing the prior belief about the weights of the regression model
     *
     * @param means An array of means of the gaussian distribution describing the prior belief about the regression weights
     * @param sigmas An array of sigma parameters of the gaussian distribution describing the prior belief about the regression weights
     */
    public LinearRidgeRegressionModelBuilder setPriorOnWeights(double[] means, double[] sigmas) {
        RegressionWeights.checkArrayHasCorrectNumberOfFeatures(means, getFeatureCount());
        RegressionWeights.checkArrayHasCorrectNumberOfFeatures(sigmas, getFeatureCount());

        this.priorOnWeightsMeans = means;
        this.priorOnWeightsSigmas = sigmas;

        return this;
    }

    /**
     * Set the input parameters to the gaussian distribution describing the prior belief about the intercept of the regression model
     *
     * @param mean The mean of the gaussian distribution describing the prior belief about the regression intercept
     * @param sigma The sigma parameter of the gaussian distribution describing the prior belief about the regression intercept
     */
    public LinearRidgeRegressionModelBuilder setPriorOnIntercept(double mean, double sigma) {
        this.priorOnInterceptMean = mean;
        this.priorOnInterceptSigma = sigma;
        return this;
    }

    /**
     * Set the input parameters to the gaussian distribution describing the prior belief about both the intercept and weights of the regression model
     *
     * @param mean The mean of the gaussian distribution describing the prior belief about both the regression intercept and weights
     * @param sigma The sigma parameter of the gaussian distribution describing the prior belief about both the regression intercept and weights
     */
    public LinearRidgeRegressionModelBuilder setPriorOnWeightsAndIntercept(double mean, double sigma) {
        setPriorOnWeights(RegressionWeights.fillPriorOnWeights(this.inputTrainingData.getShape(), mean), RegressionWeights.fillPriorOnWeights(this.inputTrainingData.getShape(), sigma));
        setPriorOnIntercept(mean, sigma);
        return this;
    }

    public LinearRidgeRegressionModelBuilder setObservationSigma(double sigma) {
        this.observationSigma = sigma;
        return this;
    }

    /**
     * Builds and fits LinearRegressionModel using the data and priors passed to the builder.
     * The model is fit using the Maximum A Posteriori algorithm, and uses ridge regression to regularize the weights.
     */
    public LinearRegressionModel build() {
        if (inputTrainingData == null || outputTrainingData == null) {
            throw new IllegalArgumentException("You have not provided both the input and output variables");
        }

        if (priorOnWeightsMeans == null || priorOnWeightsSigmas == null) {
            setPriorOnWeights(RegressionWeights.fillPriorOnWeights(this.inputTrainingData.getShape(), DEFAULT_MU), RegressionWeights.fillPriorOnWeights(this.inputTrainingData.getShape(), DEFAULT_SIGMA));
        }

        if (priorOnInterceptMean == null || priorOnInterceptSigma == null) {
            setPriorOnIntercept(DEFAULT_MU, DEFAULT_SIGMA);
        }

        DoubleVertex interceptVertex = new GaussianVertex(priorOnInterceptMean, priorOnInterceptSigma);
        DoubleVertex weightsVertex = new GaussianVertex(new long[]{1, getFeatureCount()}, ConstantVertex.of(priorOnWeightsMeans), ConstantVertex.of(priorOnWeightsSigmas));
        LinearRegressionGraph<DoubleTensor> regressionGraph = new LinearRegressionGraph<>(
            this.inputTrainingData.getShape(),
            LinearRegressionModel.gaussianOutputTransform(observationSigma),
            interceptVertex,
            weightsVertex
        );

        MAPModelFitter<DoubleTensor, DoubleTensor> fitter = new MAPModelFitter<>(regressionGraph);
        fitter.fit(inputTrainingData, outputTrainingData);
        return new LinearRegressionModel(regressionGraph);
    }

    private long getFeatureCount() {
            return this.inputTrainingData.getShape()[0];
        }
}
