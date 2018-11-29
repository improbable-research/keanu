package io.improbable.keanu.model.regression;

import io.improbable.keanu.model.ModelFitter;
import io.improbable.keanu.model.SamplingModelFitting;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

import java.util.function.Function;

/**
 * Builder class for doing linear regression without regularisation.
 *
 * @see RegressionModel
 */
public class RegressionModelBuilder<OUTPUT extends Tensor> {

    private static final double DEFAULT_MU = 0.0;
    private static final double DEFAULT_SCALE_PARAMETER = 1.0;

    private final DoubleTensor inputTrainingData;
    private final OUTPUT outputTrainingData;
    private final Function<DoubleVertex, LinearRegressionGraph.OutputVertices<OUTPUT>> outputTransform;

    private RegressionRegularization regularization = RegressionRegularization.NONE;
    private DoubleVertex priorOnWeightsScaleParameters;
    private DoubleVertex priorOnWeightsMeans;
    private DoubleVertex priorOnInterceptScaleParameter;
    private DoubleVertex priorOnInterceptMean;
    private SamplingModelFitting samplingAlgorithm = null;

    public RegressionModelBuilder(DoubleTensor inputTrainingData, OUTPUT outputTrainingData, Function<DoubleVertex, LinearRegressionGraph.OutputVertices<OUTPUT>> outputTransform) {
        this.inputTrainingData = reshapeToMatrix(inputTrainingData);
        this.outputTrainingData = reshapeToMatrix(outputTrainingData);
        this.outputTransform = outputTransform;
    }

    public RegressionModelBuilder withRegularization(RegressionRegularization regularization) {
        this.regularization = regularization;
        return this;
    }

    private static <T extends Tensor> T reshapeToMatrix(T data) {
        if (data.getRank() == 0) {
            return (T) data.reshape(1, 1);
        } else if (data.getRank() == 1) {
            return (T) data.reshape(1, data.getShape()[0]);
        } else {
            return data;
        }
    }

    /**
     * Set the input parameters to the distribution describing the prior belief about the weights of the regression model
     *
     * @param means           An array of means of the distribution describing the prior belief about the regression weights
     * @param scaleParameters An array of scale parameters of the distribution describing the prior belief about the regression weights.
     *                        This will represent sigmas if no or ridge regularization is used and will represent betas if lasso regularization is used.
     * @return this
     */
    public RegressionModelBuilder withPriorOnWeights(DoubleVertex means, DoubleVertex scaleParameters) {
        this.priorOnWeightsMeans = means;
        this.priorOnWeightsScaleParameters = scaleParameters;
        return this;
    }

    public RegressionModelBuilder withPriorOnWeights(DoubleTensor means, DoubleTensor scaleParameters) {
        return withPriorOnWeights(ConstantVertex.of(means), ConstantVertex.of(scaleParameters));
    }

    public RegressionModelBuilder withPriorOnWeights(double means, double scaleParameters) {
        return withPriorOnWeights(
            DoubleTensor.create(new double[]{means}, 1, 1),
            DoubleTensor.create(new double[]{scaleParameters}, 1, 1)
        );
    }

    /**
     * Set the input parameters to the distribution describing the prior belief about the intercept of the regression model
     *
     * @param mean           The mean of the distribution describing the prior belief about the regression intercept
     * @param scaleParameter The scale parameter of the distribution describing the prior belief about the regression intercept.
     *                       This will represent sigmas if no or ridge regularization is used and will represent betas if lasso regularization is used.
     * @return this
     */
    public RegressionModelBuilder withPriorOnIntercept(DoubleVertex mean, DoubleVertex scaleParameter) {
        this.priorOnInterceptMean = mean;
        this.priorOnInterceptScaleParameter = scaleParameter;
        return this;
    }

    public RegressionModelBuilder withPriorOnIntercept(DoubleTensor mean, DoubleTensor scaleParameter) {
        return withPriorOnIntercept(ConstantVertex.of(mean), ConstantVertex.of(scaleParameter));
    }

    public RegressionModelBuilder withPriorOnIntercept(double mean, double scaleParameter) {
        return withPriorOnIntercept(ConstantVertex.of(mean), ConstantVertex.of(scaleParameter));
    }

    /**
     * Set the input parameters to the distribution describing the prior belief about both the intercept and weights of the regression model
     *
     * @param mean           The mean of the distribution describing the prior belief about both the regression intercept and weights
     * @param scaleParameter The scale parameter of the distribution describing the prior belief about both regression intercept and weights.
     *                       This will represent sigmas if no or ridge regularization is used and will represent betas if lasso regularization is used.
     * @return this
     */
    public RegressionModelBuilder withPriorOnWeightsAndIntercept(double mean, double scaleParameter) {
        withPriorOnWeights(mean, scaleParameter);
        withPriorOnIntercept(mean, scaleParameter);
        return this;
    }

    /**
     * Optional - use a sampling algorithm to fit the model instead of the default, which is gradient optimization.
     *
     * @param sampling Defines the number of samples to take and the algorithm to use, e.g. {@link io.improbable.keanu.algorithms.mcmc.MetropolisHastings}
     * @return this
     */
    public RegressionModelBuilder withSampling(SamplingModelFitting sampling) {
        this.samplingAlgorithm = sampling;
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

        ModelFitter<DoubleTensor, OUTPUT> fitter = samplingAlgorithm == null ?
            this.regularization.createFitterForGraph(regressionGraph) :
            samplingAlgorithm.createFitterForGraph(regressionGraph);

        regressionGraph.observeValues(inputTrainingData, outputTrainingData);
        return new RegressionModel<>(regressionGraph, fitter);
    }

    private void checkVariablesAreCorrectlyInitialised() {
        if (inputTrainingData == null) {
            throw new IllegalArgumentException("You have not provided input training data");
        }
        if (outputTrainingData == null) {
            throw new IllegalArgumentException("You have not provided output training data");
        }
        if (priorOnWeightsMeans == null || priorOnWeightsScaleParameters == null) {
            withPriorOnWeights(DEFAULT_MU, DEFAULT_SCALE_PARAMETER);
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

    private long getFeatureCount() {
        return this.inputTrainingData.getShape()[0];
    }

}
