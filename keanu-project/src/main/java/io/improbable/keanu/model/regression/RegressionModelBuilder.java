package io.improbable.keanu.model.regression;

import io.improbable.keanu.model.MAPModelFitter;
import io.improbable.keanu.model.MaximumLikelihoodModelFitter;
import io.improbable.keanu.model.ModelFitter;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.LaplaceVertex;

/**
 * Builder class for doing linear regression without regularisation.
 *
 * @see RegressionModel
 */
public class RegressionModelBuilder {
    private static final double DEFAULT_MU = 0.0;
    private static final double DEFAULT_SCALE_PARAMETER = 2.0;

    private RegressionRegularization regularization = RegressionRegularization.NONE;
    private double observationSigma = DEFAULT_SCALE_PARAMETER;
    private double[] priorOnWeightsScaleParameters;
    private double[] priorOnWeightsMeans;
    private Double priorOnInterceptScaleParameter;
    private Double priorOnInterceptMean;
    private DoubleTensor inputTrainingData;
    private DoubleTensor outputLinearTrainingData;
    private BooleanTensor outputLogisticTrainingData;

    public RegressionModelBuilder setObservationSigma(double sigma) {
        this.observationSigma = sigma;
        return this;
    }

    public RegressionModelBuilder setInputTrainingData(DoubleTensor inputTrainingData) {
        this.inputTrainingData = inputTrainingData;
        return this;
    }

    /**
     * Sets the output training data for the model. Specifically the case where we are performing linear regression.
     *
     * @param outputTrainingData The output training data for the model.
     *                           This is specifically for the case where we are performing linear regression
     * @return An updated version of this builder
     */
    public RegressionModelBuilder setOutputTrainingData(DoubleTensor outputTrainingData) {
        this.outputLinearTrainingData = outputTrainingData;
        return this;
    }

    /**
     * Sets the output training data for the model. Specifically the case where we are performing logistic regression.
     *
     * @param outputTrainingData The output training data for the model.
     *                           This is specifically for the case where we are performing logistic regression
     * @return An updated version of this builder
     */
    public RegressionModelBuilder setOutputTrainingData(BooleanTensor outputTrainingData) {
        this.outputLogisticTrainingData = outputTrainingData;
        return this;
    }

    /**
     * Sets the regularization to be used by the model.
     * This can be any of the types from {@link io.improbable.keanu.model.regression.RegressionRegularization RegressionRegularization}
     *
     * @param regularization the type of regularization to be used by the resulting model
     * @return An updated version of this builder
     */
    public RegressionModelBuilder setRegularization(RegressionRegularization regularization) {
        this.regularization = regularization;
        return this;
    }

    /**
     * Set the input parameters to the distribution describing the prior belief about the weights of the regression model
     *
     * @param means An array of means of the distribution describing the prior belief about the regression weights
     * @param scaleParameters An array of scale parameters of the distribution describing the prior belief about the regression weights.
     *                        This will represent sigmas if no or ridge regularization is used and will represent betas if lasso regularization is used.
     * @return An updated version of this builder
     */
    public RegressionModelBuilder setPriorOnWeights(double[] means, double[] scaleParameters) {
        RegressionWeights.checkArrayHasCorrectNumberOfFeatures(means, getFeatureCount());
        RegressionWeights.checkArrayHasCorrectNumberOfFeatures(scaleParameters, getFeatureCount());

        this.priorOnWeightsMeans = means;
        this.priorOnWeightsScaleParameters = scaleParameters;

        return this;
    }

    /**
     * Set the input parameters to the distribution describing the prior belief about the intercept of the regression model
     *
     * @param mean The mean of the distribution describing the prior belief about the regression intercept
     * @param scaleParameter The scale parameter of the distribution describing the prior belief about the regression intercept.
     *                       This will represent sigmas if no or ridge regularization is used and will represent betas if lasso regularization is used.
     * @return An updated version of this builder
     */
    public RegressionModelBuilder setPriorOnIntercept(double mean, double scaleParameter) {
        this.priorOnInterceptMean = mean;
        this.priorOnInterceptScaleParameter = scaleParameter;
        return this;
    }

    /**
     * Set the input parameters to the distribution describing the prior belief about both the intercept and weights of the regression model
     *
     * @param mean The mean of the distribution describing the prior belief about both the regression intercept and weights
     * @param scaleParameter The scale parameter of the distribution describing the prior belief about both regression intercept and weights.
     *                       This will represent sigmas if no or ridge regularization is used and will represent betas if lasso regularization is used.
     * @return An updated version of this builder
     */
    public RegressionModelBuilder setPriorOnWeightsAndIntercept(double mean, double scaleParameter) {
        setPriorOnWeights(RegressionWeights.fillPriorOnWeights(this.inputTrainingData.getShape(), mean), RegressionWeights.fillPriorOnWeights(this.inputTrainingData.getShape(), scaleParameter));
        setPriorOnIntercept(mean, scaleParameter);
        return this;
    }

    /**
     * Builds and fits LinearRegressionModel using the data passed to the builder.
     * @return A linear regression model from the data passed to the builder
     */
    public RegressionModel<DoubleTensor> buildLinearRegressionModel() {
        checkVariablesAreCorrectlyInitialised();

        RegressionGraph<DoubleTensor> regressionGraph = new RegressionGraph<>(
            this.inputTrainingData.getShape(),
            RegressionModel.gaussianOutputTransform(observationSigma),
            getInterceptVertex(),
            getWeightsVertex()
        );

        performDataFitting(regressionGraph, outputLinearTrainingData);
        return new RegressionModel<>(regressionGraph);
    }

    /**
     * Builds and fits LogisticRegressionModel using the data passed to the builder.
     * @return A linear regression model from the data passed to the builder
     */
    public RegressionModel<BooleanTensor> buildLogisticRegressionModel() {
        checkVariablesAreCorrectlyInitialised();

        RegressionGraph<BooleanTensor> regressionGraph = new RegressionGraph<>(
                this.inputTrainingData.getShape(),
                RegressionModel.logisticOutputTransform(),
                getInterceptVertex(),
                getWeightsVertex()
        );

        performDataFitting(regressionGraph, outputLogisticTrainingData);
        return new RegressionModel<>(regressionGraph);
    }

    private void checkVariablesAreCorrectlyInitialised() {
        if (inputTrainingData == null || (outputLogisticTrainingData == null && outputLinearTrainingData == null)) {
            throw new IllegalArgumentException("You have not provided both the input and output variables");
        }
        if (priorOnWeightsMeans == null || priorOnWeightsScaleParameters == null) {
            setPriorOnWeights(RegressionWeights.fillPriorOnWeights(this.inputTrainingData.getShape(), DEFAULT_MU), RegressionWeights.fillPriorOnWeights(this.inputTrainingData.getShape(), DEFAULT_SCALE_PARAMETER));
        }

        if (priorOnInterceptMean == null || priorOnInterceptScaleParameter == null) {
            setPriorOnIntercept(DEFAULT_MU, DEFAULT_SCALE_PARAMETER);
        }
    }

    private DoubleVertex getInterceptVertex() {
        switch (this.regularization) {
            case NONE:
                return new GaussianVertex(DEFAULT_MU, DEFAULT_SCALE_PARAMETER);
            case LASSO:
                return new LaplaceVertex(priorOnInterceptMean, priorOnInterceptScaleParameter);
            case RIDGE:
                return new GaussianVertex(priorOnInterceptMean, priorOnInterceptScaleParameter);
            default:
                return new GaussianVertex(DEFAULT_MU, DEFAULT_SCALE_PARAMETER);

        }
    }

    private DoubleVertex getWeightsVertex() {
        switch (this.regularization) {
            case NONE:
                return new GaussianVertex(new long[]{1, getFeatureCount()}, ConstantVertex.of(DEFAULT_MU), ConstantVertex.of(DEFAULT_SCALE_PARAMETER));
            case LASSO:
                return new LaplaceVertex(new long[]{1, getFeatureCount()}, ConstantVertex.of(priorOnWeightsMeans), ConstantVertex.of(priorOnInterceptScaleParameter));
            case RIDGE:
                return new GaussianVertex(new long[]{1, getFeatureCount()}, ConstantVertex.of(priorOnWeightsMeans), ConstantVertex.of(priorOnInterceptScaleParameter));
            default:
                return new GaussianVertex(new long[]{1, getFeatureCount()}, ConstantVertex.of(DEFAULT_MU), ConstantVertex.of(DEFAULT_SCALE_PARAMETER));
        }
    }

    private <OUTPUT> void performDataFitting(RegressionGraph<OUTPUT> regressionGraph, OUTPUT outputTrainingData) {
        ModelFitter<DoubleTensor, OUTPUT> fitter;
        switch (this.regularization) {
            case NONE:
                fitter = new MaximumLikelihoodModelFitter<>(regressionGraph);
                fitter.fit(inputTrainingData, outputTrainingData);
                break;
            default:
                fitter = new MAPModelFitter<>(regressionGraph);
                fitter.fit(inputTrainingData, outputTrainingData);
                break;
        }
    }

    private long getFeatureCount() {
        return this.inputTrainingData.getShape()[0];
    }
}
