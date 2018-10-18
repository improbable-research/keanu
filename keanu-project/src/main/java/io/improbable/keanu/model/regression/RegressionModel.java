package io.improbable.keanu.model.regression;

import java.util.function.Function;

import io.improbable.keanu.model.Model;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

/**
 * A general linear regression model that can be fitted to input and output training data.
 * You can construct a regression model as follows:
 * <pre>
 * LinearRegressionModel model = LinearRegressionModel.builder()
 *     .setInputTrainingData(inputTrainingData)
 *     .setOutputTrainingData(outputTrainingData)
 *     .build();
 * </pre>
 */
public class RegressionModel<OUTPUT> implements Model<DoubleTensor, OUTPUT> {
    private final RegressionGraph<OUTPUT> modelGraph;

    RegressionModel(RegressionGraph<OUTPUT> modelGraph) {
        this.modelGraph = modelGraph;
    }

    public static RegressionModelBuilder builder() {
        return new RegressionModelBuilder();
    }

    static Function<DoubleVertex, RegressionGraph.OutputVertices<DoubleTensor>> gaussianOutputTransform(double measurementSigma) {
        return yVertex -> new RegressionGraph.OutputVertices<>(yVertex, new GaussianVertex(yVertex, measurementSigma));
    }

    static Function<DoubleVertex, RegressionGraph.OutputVertices<BooleanTensor>> logisticOutputTransform() {
        return probabilities -> {
            DoubleVertex sigmoid = probabilities.sigmoid();
            return new RegressionGraph.OutputVertices<>(sigmoid.greaterThan(ConstantVertex.of(0.5)), new BernoulliVertex(sigmoid));
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
