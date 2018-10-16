package io.improbable.keanu.model.regression;

import java.util.function.Function;

import io.improbable.keanu.model.Model;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

public class LinearRegressionModel implements Model<DoubleTensor, DoubleTensor> {
    private final LinearRegressionGraph<DoubleTensor> linearModelGraph;

    LinearRegressionModel(LinearRegressionGraph<DoubleTensor> linearModelGraph) {
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
        return linearModelGraph.getX();
    }

    public DoubleTensor getWeights() {
        return linearModelGraph.getWeights();
    }

    public double getIntercept() {
        return linearModelGraph.getIntercept();
    }

    public double getWeight(int index) {
        DoubleTensor weight = linearModelGraph.getWeights();
        return weight.isScalar() ? weight.scalar() : weight.getValue(0, index);
    }

    @Override
    public DoubleTensor predict(DoubleTensor tensor) {
        return linearModelGraph.predict(tensor);
    }
}
