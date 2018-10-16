package io.improbable.keanu.model.regression;


import java.util.function.Function;

import io.improbable.keanu.model.BayesianModel;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

public class LogisticRegressionModel extends BayesianModel<DoubleTensor, BooleanTensor> {

    private final LinearRegressionGraph linearModelGraph;

    LogisticRegressionModel(LinearRegressionGraph<BooleanTensor> linearModelGraph) {
        super(linearModelGraph);
        this.linearModelGraph = linearModelGraph;
    }

    public static LogisticRegressionModelBuilder builder() {
        return new LogisticRegressionModelBuilder();
    }

    public static LogisticLassoRegressionModelBuilder lassoRegressionModelBuilder() {
        return new LogisticLassoRegressionModelBuilder();
    }

    public static LogisticRidgeRegressionModelBuilder ridgeRegressionModelBuilder() {
        return new LogisticRidgeRegressionModelBuilder();
    }

    static Function<DoubleVertex, LinearRegressionGraph.OutputVertices<BooleanTensor>> logisticOutputTransform() {
        return probabilities -> {
            DoubleVertex sigmoid = probabilities.sigmoid();
            return new LinearRegressionGraph.OutputVertices<>(sigmoid.greaterThan(ConstantVertex.of(0.5)), new BernoulliVertex(sigmoid));
        };
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

}
