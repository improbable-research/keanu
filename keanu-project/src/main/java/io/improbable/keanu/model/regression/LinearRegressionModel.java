package io.improbable.keanu.model.regression;

import java.util.function.Function;

import io.improbable.keanu.model.BayesianModel;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

public class LinearRegressionModel extends BayesianModel<DoubleTensor, DoubleTensor> {
    private final LinearRegressionGraph linearModelGraph;

    LinearRegressionModel(LinearRegressionGraph<DoubleTensor> linearModelGraph) {
        super(linearModelGraph);
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

    static Function<DoubleVertex, LinearRegressionGraph.OutputVertices<DoubleTensor>> gaussianOutputTransform(double measurementSigma) {
        return yVertex -> new LinearRegressionGraph.OutputVertices<>(yVertex, new GaussianVertex(yVertex, measurementSigma));
    }

    public DoubleTensor getY() {
        return linearModelGraph.getXVertex().getValue();
    }

    public DoubleTensor getWeights() {
        return linearModelGraph.getWeightsVertex().getValue();
    }

    public double getIntercept() {
        return linearModelGraph.getInterceptVertex().getValue().scalar();
    }

    public double getWeight(int index) {
        DoubleVertex weight = linearModelGraph.getWeightsVertex();
        return weight.getValue().isScalar() ? weight.getValue().scalar() : weight.getValue(0, index);
    }
}
