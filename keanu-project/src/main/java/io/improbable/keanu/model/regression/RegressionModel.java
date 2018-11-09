package io.improbable.keanu.model.regression;

import io.improbable.keanu.model.Model;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

import java.util.function.Function;

/**
 * A general linear regression model that can be fitted to input and output training data.
 * You can construct a regression model as follows:
 * <pre>
 * RegressionModel model = RegressionModel
 *      .withTrainingData(inputTrainingData, outputTrainingData)
 *      .withRegularization(RegressionRegularization.RIDGE)
 *      .build();
 * </pre>
 */
public class RegressionModel<OUTPUT> implements Model<DoubleTensor, OUTPUT> {
    private static final double DEFAULT_OBSERVATION_SIGMA = 1.0;

    private final LinearRegressionGraph<OUTPUT> modelGraph;

    RegressionModel(LinearRegressionGraph<OUTPUT> modelGraph) {
        this.modelGraph = modelGraph;
    }

    public static RegressionModelBuilder<DoubleTensor> withTrainingData(DoubleTensor inputTrainingData, DoubleTensor outputTrainingData) {
        return new RegressionModelBuilder<>(inputTrainingData, outputTrainingData, RegressionModel.gaussianOutputTransform(DEFAULT_OBSERVATION_SIGMA));
    }

    public static RegressionModelBuilder<BooleanTensor> withTrainingData(DoubleTensor inputTrainingData, BooleanTensor outputTrainingData) {
        return new RegressionModelBuilder<>(inputTrainingData, outputTrainingData, RegressionModel.logisticOutputTransform());
    }

    static Function<DoubleVertex, LinearRegressionGraph.OutputVertices<DoubleTensor>> gaussianOutputTransform(double measurementSigma) {
        return yVertex -> new LinearRegressionGraph.OutputVertices<>(yVertex, new GaussianVertex(yVertex, measurementSigma));
    }

    static Function<DoubleVertex, LinearRegressionGraph.OutputVertices<BooleanTensor>> logisticOutputTransform() {
        return probabilities -> {
            DoubleVertex sigmoid = probabilities.sigmoid();
            return new LinearRegressionGraph.OutputVertices<>(sigmoid.greaterThan(ConstantVertex.of(0.5)), new BernoulliVertex(sigmoid));
        };
    }

    public DoubleTensor getWeights() {
        return modelGraph.getWeights();
    }

    public double getIntercept() {
        return modelGraph.getIntercept();
    }

    public double getWeight(int index) {
        return getWeights().getFlattenedView().getOrScalar(index);
    }

    @Override
    public OUTPUT predict(DoubleTensor tensor) {
        return modelGraph.predict(tensor);
    }
}
