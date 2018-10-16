package io.improbable.keanu.model.regression;

import io.improbable.keanu.model.MAPModelFitter;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.LaplaceVertex;
import lombok.experimental.UtilityClass;

/**
 * Class for building a logistic regression model.
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

    public LogisticLassoRegressionModelBuilder setPriorOnWeights(double[] means, double[] betas) {
        RegressionWeights.checkLaplaceParameters(getFeatureCount(), means, betas);

        this.priorOnWeightsMeans = means;
        this.priorOnWeightsBetas = betas;

        return this;
    }

    public LogisticLassoRegressionModelBuilder setPriorOnIntercept(double mean, double beta) {
        this.priorOnInterceptMean = mean;
        this.priorOnInterceptBeta = beta;
        return this;
    }

    public LogisticLassoRegressionModelBuilder setPriorOnWeightsAndIntercept(double mean, double beta) {
        setPriorOnWeights(RegressionWeights.fillPriorOnWeights(this.inputTrainingData.getShape(), mean), RegressionWeights.fillPriorOnWeights(this.inputTrainingData.getShape(), beta));
        setPriorOnIntercept(mean, beta);
        return this;
    }

    public LogisticRegressionModel build() {
        DoubleVertex interceptVertex = new LaplaceVertex(priorOnInterceptMean, priorOnInterceptBeta);
        DoubleVertex weightsVertex = new LaplaceVertex(new long[]{1, getFeatureCount()}, ConstantVertex.of(priorOnWeightsMeans), ConstantVertex.of(priorOnWeightsBetas));
        LinearRegressionGraph<BooleanTensor> regressionGraph = new LinearRegressionGraph<>(
            this.inputTrainingData.getShape(),
            LogisticRegressionModel.logisticOutputTransform(),
            interceptVertex,
            weightsVertex
        );

        return new LogisticRegressionModel(inputTrainingData, outputTrainingData, regressionGraph, new MAPModelFitter<>());
    }

    private long getFeatureCount() {
            return this.inputTrainingData.getShape()[0];
        }
}
