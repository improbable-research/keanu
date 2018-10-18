package io.improbable.keanu.model.regression;

import io.improbable.keanu.model.MaximumLikelihoodModelFitter;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

/**
 * Builder class for building a logistic regression model without regularisation.
 *
 * @see LogisticRegressionModel
 */
public class LogisticRegressionModelBuilder {
    private static final double DEFAULT_MU = 0.0;
    private static final double DEFAULT_SIGMA = 2.0;

    private DoubleTensor inputTrainingData;
    private BooleanTensor outputTrainingData;

    public static LogisticRegressionModelBuilder withFeatureShape() {
        return new LogisticRegressionModelBuilder();
    }

    public LogisticRegressionModelBuilder setInputTrainingData(DoubleTensor inputTrainingData) {
        this.inputTrainingData = inputTrainingData;
        return this;
    }

    public LogisticRegressionModelBuilder setOutputTrainingData(BooleanTensor outputTrainingData) {
        this.outputTrainingData = outputTrainingData;
        return this;
    }

    /**
     * Builds and fits LogisticRegressionModel using the data passed to the builder.
     * The model is fit using the Maximum Likelihood algorithm and no regularization is performed on the weights.
     */
    public LogisticRegressionModel build() {
        DoubleVertex interceptVertex = new GaussianVertex(DEFAULT_MU, DEFAULT_SIGMA);
        DoubleVertex weightsVertex = new GaussianVertex(new long[]{1, getFeatureCount()}, ConstantVertex.of(DEFAULT_MU), ConstantVertex.of(DEFAULT_SIGMA));
        RegressionGraph<BooleanTensor> regressionGraph = new RegressionGraph<>(
            this.inputTrainingData.getShape(),
            LogisticRegressionModel.logisticOutputTransform(),
            interceptVertex,
            weightsVertex
        );

        MaximumLikelihoodModelFitter<DoubleTensor, BooleanTensor> fitter = new MaximumLikelihoodModelFitter<>(regressionGraph);
        fitter.fit(inputTrainingData, outputTrainingData);
        return new LogisticRegressionModel(regressionGraph);
    }

    private long getFeatureCount() {
            return this.inputTrainingData.getShape()[0];
        }
}
