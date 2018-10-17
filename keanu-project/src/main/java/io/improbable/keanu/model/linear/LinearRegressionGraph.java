package io.improbable.keanu.model.linear;

import static io.improbable.keanu.model.linear.LinearModel.INTERCEPT_LABEL;
import static io.improbable.keanu.model.linear.LinearModel.WEIGHTS_LABEL;
import static io.improbable.keanu.model.linear.LinearModel.X_LABEL;
import static io.improbable.keanu.model.linear.LinearModel.Y_LABEL;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

public class LinearRegressionGraph {

    private LinearRegressionGraph() {
    }

    public static BayesianNetwork build(DoubleTensor x, double priorOnMu, double priorOnSigma, double[] priorOnSigmaForWeights) {
        long numberOfFeatures = x.getShape()[0];
        long[] weightShape = new long[]{1, numberOfFeatures};
        DoubleVertex weights = new GaussianVertex(weightShape, priorOnMu, ConstantVertex.of(priorOnSigmaForWeights)).setLabel(WEIGHTS_LABEL);
        DoubleVertex intercept = new GaussianVertex(priorOnMu, priorOnSigma).setLabel(INTERCEPT_LABEL);
        DoubleVertex xVertex = ConstantVertex.of(x).setLabel(X_LABEL);
        DoubleVertex yVertex = weights.getValue().isScalar() ?
            weights.times(xVertex).plus(intercept).setLabel(Y_LABEL) :
            weights.matrixMultiply(xVertex).plus(intercept).setLabel(Y_LABEL);
        return new BayesianNetwork(yVertex.getConnectedGraph());
    }

}
