package io.improbable.keanu.model.regression;

import io.improbable.keanu.model.MAPModelFitter;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.LaplaceVertex;

/**
 * Builder class for creating a logistic lasso regression model.
 * <p>
 * This builds <a href="https://en.wikipedia.org/wiki/Lasso_(statistics)">lasso</a> logistic regression, meaning that
 * the weights have a Laplace prior distribution, i.e. the model has <a href="http://mathworld.wolfram.com/L1-Norm.html">L1 norm regularisation</a>.
 *
 * @see LogisticRegressionModel
 */
public class LogisticLassoRegressionModelBuilder {
    private static final double DEFAULT_MU = 0.0;
    private static final double DEFAULT_BETA = 2.0;

    private DoubleTensor inputTrainingData;
    private BooleanTensor outputTrainingData;

    double[] priorOnWeightsBetas;
    double[] priorOnWeightsMeans;
    double priorOnInterceptMean;
    double priorOnInterceptBeta;

    LogisticLassoRegressionModelBuilder() {
        setPriorOnWeightsAndIntercept(DEFAULT_MU, DEFAULT_BETA);
    }

    public LogisticLassoRegressionModelBuilder setInputTrainingData(DoubleTensor inputTrainingData) {
        this.inputTrainingData = inputTrainingData;
        return this;
    }

    public LogisticLassoRegressionModelBuilder setOuputTrainingData(BooleanTensor outputTrainingData) {
        this.outputTrainingData = outputTrainingData;
        return this;
    }

    /**
     * Set the input parameters to the Laplace distribution describing the prior belief about the weights of the regression model
     *
     * @param means An array of means of the laplace distribution describing the prior belief about the regression weights
     * @param betas An array of beta parameters of the laplace distribution describing the prior belief about the regression weights
     */
    public LogisticLassoRegressionModelBuilder setPriorOnWeights(double[] means, double[] betas) {
        RegressionWeights.checkLaplaceParameters(getFeatureCount(), means, betas);

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
    public LogisticLassoRegressionModelBuilder setPriorOnIntercept(double mean, double beta) {
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
    public LogisticLassoRegressionModelBuilder setPriorOnWeightsAndIntercept(double mean, double beta) {
        setPriorOnWeights(RegressionWeights.fillPriorOnWeights(this.inputTrainingData.getShape(), mean), RegressionWeights.fillPriorOnWeights(this.inputTrainingData.getShape(), beta));
        setPriorOnIntercept(mean, beta);
        return this;
    }


    /**
     * Builds and fits LogisticRegressionModel using the data and priors passed to the builder.
     * The model is fit using the Maximum A Posteriori algorithm and lasso regularization is performed on the weights.
     */
    public LogisticRegressionModel build() {
        DoubleVertex interceptVertex = new LaplaceVertex(priorOnInterceptMean, priorOnInterceptBeta);
        DoubleVertex weightsVertex = new LaplaceVertex(new long[]{1, getFeatureCount()}, ConstantVertex.of(priorOnWeightsMeans), ConstantVertex.of(priorOnWeightsBetas));
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
