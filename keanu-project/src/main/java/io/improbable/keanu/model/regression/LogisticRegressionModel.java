package io.improbable.keanu.model.regression;


import java.util.function.Function;

import io.improbable.keanu.model.Model;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

/**
 * A general logistic regression model that can be fitted to input and output training data.
 * You can construct a regression model as follows:
 * <pre>
 * LogisticRegressionModel model = LogisticRegressionModel.builder()
 *     .setInputTrainingData(inputTrainingData)
 *     .setOutputTrainingData(outputTrainingData)
 *     .build();
 * </pre>
 */
public class LogisticRegressionModel implements Model<DoubleTensor, BooleanTensor> {

    private final RegressionGraph<BooleanTensor> linearModelGraph;

    LogisticRegressionModel(RegressionGraph<BooleanTensor> linearModelGraph) {
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

    static Function<DoubleVertex, RegressionGraph.OutputVertices<BooleanTensor>> logisticOutputTransform() {
        return probabilities -> {
            DoubleVertex sigmoid = probabilities.sigmoid();
            return new RegressionGraph.OutputVertices<>(sigmoid.greaterThan(ConstantVertex.of(0.5)), new BernoulliVertex(sigmoid));
        };
    }

    public DoubleTensor getWeights() {
        return linearModelGraph.getWeights();
    }

    public double getIntercept() {
        return linearModelGraph.getIntercept();
    }

    @Override
    public BooleanTensor predict(DoubleTensor tensor) {
        return linearModelGraph.predict(tensor);
    }
}
