package io.improbable.keanu.model.linear;

import java.util.Arrays;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

public class LinearRegression implements LinearModel {

    private static final double DEFAULT_MU = 0.0;
    private static final double DEFAULT_SIGMA = 2.0;

    private final BayesianNetwork net;
    private final DoubleTensor y;

    public LinearRegression(DoubleTensor x, DoubleTensor y) {
        this(x, y, DEFAULT_MU, DEFAULT_SIGMA);
    }

    public LinearRegression(DoubleTensor x, DoubleTensor y, double priorOnMu, double priorOnSigma) {
        this(x, y, priorOnMu, priorOnSigma, fillPriorOnWeights(x, priorOnSigma));
    }

    public LinearRegression(DoubleTensor x, DoubleTensor y, double priorOnMu, double priorOnSigma, double... priorOnSigmaForWeights) {
        this.y = y;
        this.net = buildModel(x, priorOnMu, priorOnSigma, priorOnSigmaForWeights);
    }

    @Override
    public DoubleTensor getY() {
        return y;
    }

    public DoubleVertex getWeights() {
        return (DoubleVertex) net.getVertexByLabel(WEIGHTS_LABEL);
    }

    public DoubleVertex getIntercept() {
        return (DoubleVertex) net.getVertexByLabel(INTERCEPT_LABEL);
    }

    public BayesianNetwork getNet() {
        return net;
    }

    public double getWeight(int index) {
        DoubleVertex weight = (DoubleVertex) net.getVertexByLabel(WEIGHTS_LABEL);
        return weight.getValue().isScalar() ? weight.getValue().scalar() : weight.getValue(0, index);
    }

    private BayesianNetwork buildModel(DoubleTensor x, double priorOnMu, double priorOnSigma, double... priorOnSigmaForWeights) {
        BayesianNetwork net = LinearRegressionGraph.build(x, priorOnMu, priorOnSigma, priorOnSigmaForWeights);
        DoubleVertex yVertex = (DoubleVertex) net.getVertexByLabel(Y_LABEL);
        DoubleVertex yObservable = new GaussianVertex(yVertex, priorOnSigma).setLabel(Y_OBSERVATION_LABEL);
        return new BayesianNetwork(yObservable.getConnectedGraph());
    }

    private static double[] fillPriorOnWeights(DoubleTensor x, double priorOnSigma) {
        double[] priorWeights = new double[(int) x.getShape()[0]];
        Arrays.fill(priorWeights, priorOnSigma);
        return priorWeights;
    }

}
