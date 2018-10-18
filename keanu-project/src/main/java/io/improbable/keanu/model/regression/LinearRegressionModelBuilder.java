package io.improbable.keanu.model.regression;

import io.improbable.keanu.model.MaximumLikelihoodModelFitter;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

/**
 * Builder class for doing linear regression without regularisation.
 *
 * @see LinearRegressionModel
 */
public class LinearRegressionModelBuilder {
    private static final double DEFAULT_MU = 0.0;
    private static final double DEFAULT_SIGMA = 2.0;

    private double observationSigma = DEFAULT_SIGMA;
    private DoubleTensor inputTrainingData;
    private DoubleTensor outputTrainingData;

    public LinearRegressionModelBuilder setObservationSigma(double sigma) {
        this.observationSigma = sigma;
        return this;
    }

    public LinearRegressionModelBuilder setInputTrainingData(DoubleTensor inputTrainingData) {
        this.inputTrainingData = inputTrainingData;
        return this;
    }

    public LinearRegressionModelBuilder setOutputTrainingData(DoubleTensor outputTrainingData) {
        this.outputTrainingData = outputTrainingData;
        return this;
    }

    /**
     * Builds and fits LinearRegressionModel using the data passed to the builder.
     * The model is fit using the Maximum Likelihood algorithm.
     */
    public LinearRegressionModel build() {
        if (inputTrainingData == null || outputTrainingData == null) {
            throw new IllegalArgumentException("You have not provided both the input and output variables");
        }

        DoubleVertex interceptVertex = new GaussianVertex(DEFAULT_MU, DEFAULT_SIGMA);
        DoubleVertex weightsVertex = new GaussianVertex(new long[]{1, getFeatureCount()}, ConstantVertex.of(DEFAULT_MU), ConstantVertex.of(DEFAULT_SIGMA));
        RegressionGraph<DoubleTensor> regressionGraph = new RegressionGraph<>(
            this.inputTrainingData.getShape(),
            LinearRegressionModel.gaussianOutputTransform(observationSigma),
            interceptVertex,
            weightsVertex
        );

        MaximumLikelihoodModelFitter<DoubleTensor, DoubleTensor> fitter = new MaximumLikelihoodModelFitter<>(regressionGraph);
        fitter.fit(inputTrainingData, outputTrainingData);
        return new LinearRegressionModel(regressionGraph);
    }

    private long getFeatureCount() {
        return this.inputTrainingData.getShape()[0];
    }
}
