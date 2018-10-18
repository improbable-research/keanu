package io.improbable.keanu.model.regression;

import java.util.function.Function;

import io.improbable.keanu.model.Model;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
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
public class LinearRegressionModel implements Model<DoubleTensor, DoubleTensor> {
    private final RegressionGraph<DoubleTensor> linearModelGraph;

    LinearRegressionModel(RegressionGraph<DoubleTensor> linearModelGraph) {
        this.linearModelGraph = linearModelGraph;
    }

    public static LinearRegressionModelBuilder builder() {
        return new LinearRegressionModelBuilder();
    }

    public static LinearRidgeRegressionModelBuilder ridgeRegressionModelBuilder() {
        return new LinearRidgeRegressionModelBuilder();
    }

    public static LinearLassoRegressionModelBuilder lassoRegressionModelBuilder() {
        return new LinearLassoRegressionModelBuilder();
    }

    static Function<DoubleVertex, RegressionGraph.OutputVertices<DoubleTensor>> gaussianOutputTransform(double measurementSigma) {
        return yVertex -> new RegressionGraph.OutputVertices<>(yVertex, new GaussianVertex(yVertex, measurementSigma));
    }

    public DoubleTensor getWeights() {
        return linearModelGraph.getWeights();
    }

    public double getIntercept() {
        return linearModelGraph.getIntercept();
    }

    public double getWeight(int index) {
        return getWeights().getFlattenedView().getOrScalar(index);
    }

    @Override
    public DoubleTensor predict(DoubleTensor tensor) {
        return linearModelGraph.predict(tensor);
    }
}
