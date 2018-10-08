package io.improbable.keanu.model.linear;

import io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizer;
import io.improbable.keanu.model.Model;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

public interface LinearModel extends Model {

    VertexLabel WEIGHTS_LABEL = new VertexLabel("weights");
    VertexLabel Y_OBSERVATION_LABEL = new VertexLabel("yObservation");
    VertexLabel INTERCEPT_LABEL = new VertexLabel("intercept");
    VertexLabel X_LABEL = new VertexLabel("x");
    VertexLabel Y_LABEL = new VertexLabel("y");

    default Vertex getXVertex() {
        return getNet().getVertexByLabel(X_LABEL);
    }

    default Vertex getYVertex() {
        return getNet().getVertexByLabel(Y_LABEL);
    }

    default void fit() {
        BayesianNetwork net = getNet();
        net.getVertexByLabel(Y_OBSERVATION_LABEL).observe(getY());
        GradientOptimizer optimizer = GradientOptimizer.of(net);
        optimizer.maxAPosteriori();
    }

    default DoubleTensor predict(DoubleTensor x) {
        DoubleVertex xVertex = ((DoubleVertex) getXVertex());
        xVertex.setAndCascade(x);
        return ((DoubleVertex) getYVertex()).getValue();
    }

    default double score(DoubleTensor x, DoubleTensor yTrue) {
        DoubleTensor yPredicted = predict(x);
        double residualSumOfSquares = (yTrue.minus(yPredicted).pow(2.)).sum();
        double totalSumOfSquares = ((yTrue.minus(yTrue.average())).pow(2.)).sum();
        return 1 - (residualSumOfSquares / totalSumOfSquares);
    }

    Tensor getY();

}
