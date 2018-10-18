package io.improbable.keanu.model.regression;

import io.improbable.keanu.model.MAPModelFitter;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.LaplaceVertex;

/**
 * Builder class for creating a linear lasso regression model.
 * <p>
 * This builds <a href="https://en.wikipedia.org/wiki/Lasso_(statistics)">lasso</a> linear regression, meaning that
 * the weights have a Laplace prior distribution, i.e. the model has <a href="http://mathworld.wolfram.com/L1-Norm.html">L1 norm regularisation</a>.
 *
 * @see LinearRegressionModel
 */
public class LinearLassoRegressionModelBuilder {
    private static final double DEFAULT_MU = 0.0;
    private static final double DEFAULT_BETA = 2.0;

    private DoubleTensor inputTrainingData;
    private DoubleTensor outputTrainingData;
    double[] priorOnWeightsBetas;
    double[] priorOnWeightsMeans;
    Double priorOnInterceptMean;
    Double priorOnInterceptBeta;
    double observationSigma = DEFAULT_BETA;

    public LinearLassoRegressionModelBuilder setInputTrainingData(DoubleTensor inputTrainingData) {
        this.inputTrainingData = inputTrainingData;
        return this;
    }

    public LinearLassoRegressionModelBuilder setOutputTrainingData(DoubleTensor outputTrainingData) {
        this.outputTrainingData = outputTrainingData;
        return this;
    }

    /**
     * Set the input parameters to the laplace distribution describing the prior belief about the weights of the regression model
     *
     * @param means An array of means of the laplace distribution describing the prior belief about the regression weights
     * @param betas An array of beta parameters of the laplace distribution describing the prior belief about the regression weights
     */
    public LinearLassoRegressionModelBuilder setPriorOnWeights(double[] means, double[] betas) {
        RegressionWeights.checkArrayHasCorrectNumberOfFeatures(means, getFeatureCount());
        RegressionWeights.checkArrayHasCorrectNumberOfFeatures(betas, getFeatureCount());

        this.priorOnWeightsMeans = means;
        this.priorOnWeightsBetas = betas;

        return this;
    }

    /**
     * Set the input parameters to the Laplace distribution describing the prior belief about the intercept of the regression model
     *
     * @param mean The mean of the laplace distribution describing the prior belief about the regression intercept
     * @param beta The beta parameter of the laplace distribution describing the prior belief about the regression intercept
     */
    public LinearLassoRegressionModelBuilder setPriorOnIntercept(double mean, double beta) {
        this.priorOnInterceptMean = mean;
        this.priorOnInterceptBeta = beta;
        return this;
    }

    /**
     * Set the input parameters to the Laplace distribution describing the prior belief about both the intercept and weights of the regression model
     *
     * @param mean The mean of the laplace distribution describing the prior belief about both the regression intercept and weights
     * @param beta The beta parameter of the laplace distribution describing the prior belief about both the regression intercept and weights
     */
    public LinearLassoRegressionModelBuilder setPriorOnWeightsAndIntercept(double mean, double beta) {
        setPriorOnWeights(RegressionWeights.fillPriorOnWeights(this.inputTrainingData.getShape(), mean), RegressionWeights.fillPriorOnWeights(this.inputTrainingData.getShape(), beta));
        setPriorOnIntercept(mean, beta);
        return this;
    }

    public LinearLassoRegressionModelBuilder setObservationSigma(double sigma) {
        this.observationSigma = sigma;
        return this;
    }

    /**
     * Builds and fits LinearRegressionModel using the data and priors passed to the builder.
     * The model is fit using the Maximum A Posteriori algorithm, and uses lasso regression to regularize the weights.
     */
    public LinearRegressionModel build() {
        if (inputTrainingData == null || outputTrainingData == null) {
            throw new IllegalArgumentException("You have not provided both the input and output variables");
        }

        if (priorOnWeightsMeans == null || priorOnWeightsBetas == null) {
            setPriorOnWeights(RegressionWeights.fillPriorOnWeights(this.inputTrainingData.getShape(), DEFAULT_MU), RegressionWeights.fillPriorOnWeights(this.inputTrainingData.getShape(), DEFAULT_BETA));
        }

        if (priorOnInterceptMean == null || priorOnInterceptBeta == null) {
            setPriorOnIntercept(DEFAULT_MU, DEFAULT_BETA);
        }

        DoubleVertex interceptVertex = new LaplaceVertex(priorOnInterceptMean, priorOnInterceptBeta);
        DoubleVertex weightsVertex = new LaplaceVertex(new long[]{1, getFeatureCount()}, ConstantVertex.of(priorOnWeightsMeans), ConstantVertex.of(priorOnWeightsBetas));
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
