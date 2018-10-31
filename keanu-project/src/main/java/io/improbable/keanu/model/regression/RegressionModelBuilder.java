package io.improbable.keanu.model.regression;

import io.improbable.keanu.algorithms.PosteriorSamplingAlgorithm;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.model.ModelFitter;
import io.improbable.keanu.model.SamplingModelFitter;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

import java.util.function.Function;

/**
 * Builder class for doing linear regression without regularisation.
 *
 * @see RegressionModel
 */
public class RegressionModelBuilder<OUTPUT> {
    private static final double DEFAULT_MU = 0.0;
    private static final double DEFAULT_SCALE_PARAMETER = 1.0;

    private RegressionRegularization regularization = RegressionRegularization.NONE;
    private double[] priorOnWeightsScaleParameters;
    private double[] priorOnWeightsMeans;
    private Double priorOnInterceptScaleParameter;
    private Double priorOnInterceptMean;
    private DoubleTensor inputTrainingData;
    private OUTPUT outputTrainingData;
    private Function<DoubleVertex, LinearRegressionGraph.OutputVertices<OUTPUT>> outputTransform;
    private PosteriorSamplingAlgorithm samplingAlgorithm = null;
    private int samplingCount = 10000;

    public RegressionModelBuilder(DoubleTensor inputTrainingData, OUTPUT outputTrainingData, Function<DoubleVertex, LinearRegressionGraph.OutputVertices<OUTPUT>> outputTransform) {
        this.inputTrainingData = inputTrainingData;
        this.outputTrainingData = outputTrainingData;
        this.outputTransform = outputTransform;
    }

    public RegressionModelBuilder withRegularization(RegressionRegularization regularization) {
        this.regularization = regularization;
        return this;
    }

    /**
     * Set the input parameters to the distribution describing the prior belief about the weights of the regression model
     *
     * @param means An array of means of the distribution describing the prior belief about the regression weights
     * @param scaleParameters An array of scale parameters of the distribution describing the prior belief about the regression weights.
     *                        This will represent sigmas if no or ridge regularization is used and will represent betas if lasso regularization is used.
     * @return this
     */
    public RegressionModelBuilder withPriorOnWeights(double[] means, double[] scaleParameters) {
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
     * @return this
     */
    public RegressionModelBuilder withPriorOnIntercept(double mean, double scaleParameter) {
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
     * @return this
     */
    public RegressionModelBuilder withPriorOnWeightsAndIntercept(double mean, double scaleParameter) {
        withPriorOnWeights(RegressionWeights.fillPriorOnWeights(this.inputTrainingData.getShape(), mean), RegressionWeights.fillPriorOnWeights(this.inputTrainingData.getShape(), scaleParameter));
        withPriorOnIntercept(mean, scaleParameter);
        return this;
    }

    public RegressionModelBuilder withSampling(int count) {
        return withSampling(MetropolisHastings.withDefaultConfig(), count);
    }

    public RegressionModelBuilder withSampling(PosteriorSamplingAlgorithm samplingAlgorithm, int count) {
        this.samplingAlgorithm = samplingAlgorithm;
        this.samplingCount = count;
        return this;
    }

    /**
     * @return A linear regression model from the data passed to the builder
     */
    public RegressionModel<OUTPUT> build() {
        checkVariablesAreCorrectlyInitialised();

        LinearRegressionGraph<OUTPUT> regressionGraph = new LinearRegressionGraph<>(
                this.inputTrainingData.getShape(),
                outputTransform,
                getInterceptVertex(),
                getWeightsVertex()
        );

        performDataFitting(regressionGraph, outputTrainingData);
        return new RegressionModel<>(regressionGraph);
    }

    private void checkVariablesAreCorrectlyInitialised() {
        if (inputTrainingData == null) {
            throw new IllegalArgumentException("You have not provided input training data");
        }
        if (outputTrainingData == null) {
            throw new IllegalArgumentException("You have not provided output training data");
        }
        if (priorOnWeightsMeans == null || priorOnWeightsScaleParameters == null) {
            withPriorOnWeights(RegressionWeights.fillPriorOnWeights(this.inputTrainingData.getShape(), DEFAULT_MU), RegressionWeights.fillPriorOnWeights(this.inputTrainingData.getShape(), DEFAULT_SCALE_PARAMETER));
        }
        if (priorOnInterceptMean == null || priorOnInterceptScaleParameter == null) {
            withPriorOnIntercept(DEFAULT_MU, DEFAULT_SCALE_PARAMETER);
        }
    }

    private DoubleVertex getInterceptVertex() {
        return this.regularization.getInterceptVertex(priorOnInterceptMean, priorOnInterceptScaleParameter);
    }

    private DoubleVertex getWeightsVertex() {
        return this.regularization.getWeightsVertex(getFeatureCount(), priorOnWeightsMeans, priorOnWeightsScaleParameters);
    }

    private void performDataFitting(LinearRegressionGraph<OUTPUT> regressionGraph, OUTPUT outputTrainingData) {
        ModelFitter<DoubleTensor, OUTPUT> fitter = samplingAlgorithm == null ?
            this.regularization.createFitterForGraph(regressionGraph) :
            new SamplingModelFitter<>(regressionGraph, samplingAlgorithm, samplingCount);

        fitter.fit(inputTrainingData, outputTrainingData);
    }

    private long getFeatureCount() {
        return this.inputTrainingData.getShape()[0];
    }

}
