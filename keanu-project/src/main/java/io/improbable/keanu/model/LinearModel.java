package io.improbable.keanu.model;

import io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizer;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

public interface LinearModel extends Model {

    VertexLabel WEIGHTS_LABEL = new VertexLabel("weights");
    VertexLabel Y_OBSERVATION_LABEL = new VertexLabel("yObservation");
    VertexLabel X_LABEL = new VertexLabel("x");
    VertexLabel Y_LABEL = new VertexLabel("y");

    BayesianNetwork getNet();

    default Vertex getXVertex() {
        return getNet().getVertexByLabel(X_LABEL);
    }

    default Vertex getYVertex() {
        return getNet().getVertexByLabel(Y_LABEL);
    }

    default void fit() {
        BayesianNetwork net = getNet();
        Tensor y = ((Vertex<? extends Tensor>) getYVertex()).getValue();
        net.getVertexByLabel(Y_OBSERVATION_LABEL).observe(y);
        GradientOptimizer optimizer = GradientOptimizer.of(net);
        optimizer.maxAPosteriori();
    }

    default boolean isFit() {
        return true;
    }

    default DoubleTensor predict(DoubleTensor x) {
        if (isFit()) {
            DoubleVertex xVertex = ((DoubleVertex) getXVertex());
            xVertex.setAndCascade(x);
            return ((DoubleVertex) getYVertex()).getValue();
        } else {
            throw new RuntimeException("The model must be fit before attempting to predict.");
        }
    }

    default double score(DoubleTensor x, DoubleTensor yTrue) {
        DoubleTensor yPredicted = predict(x);
        double residualSumOfSquares = (yTrue.minus(yPredicted).pow(2.)).sum();
        double totalSumOfSquares = ((yTrue.minus(yTrue.average())).pow(2.)).sum();
        return 1 - (residualSumOfSquares / totalSumOfSquares);
    }

}
