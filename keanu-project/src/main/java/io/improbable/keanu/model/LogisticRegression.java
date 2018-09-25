package io.improbable.keanu.model;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

public class LogisticRegression implements LinearModel {

    private static final double DEFAULT_MU = 0.0;
    private static final double DEFAULT_SIGMA = 2.0;
    private static final double DEFAULT_REGULARIZATION = 1;
    private static final VertexLabel WEIGHTS_LABEL = new VertexLabel("weights");
    private static final VertexLabel INTERCEPT_LABEL = new VertexLabel("intercept");
    private static final VertexLabel Y_OBSERVATION_LABEL = new VertexLabel("y");

    private final DoubleTensor y;
    private final double priorSigma;
    private final double regularization;
    private final LinearRegression regression;
    private final BayesianNetwork net;

    public LogisticRegression(DoubleTensor x, DoubleTensor y, double regularization, double priorMu, double priorSigma) {
        this.y = y;
        this.priorSigma = priorSigma;
        this.regularization = regularization;
        this.regression = new LinearRegression(x, y, priorMu, priorSigma, regularizeSigma(x));
        this.net = construct();
        this.regression.setNet(net);
    }

    public LogisticRegression(DoubleTensor x, DoubleTensor y) {
        this(x, y, DEFAULT_REGULARIZATION, DEFAULT_MU, DEFAULT_SIGMA);
    }

    public LogisticRegression(DoubleTensor x, DoubleTensor y, double regularization) {
        this(x, y, regularization, DEFAULT_MU, DEFAULT_SIGMA);
    }

    public LogisticRegression(DoubleTensor x, DoubleTensor y, double priorMu, double priorSigma) {
        this(x, y, DEFAULT_REGULARIZATION, priorMu, priorSigma);
    }

    @Override
    public BayesianNetwork construct() {
        BayesianNetwork linearNetwork = regression.getNet();
        DoubleVertex probabilities = (DoubleVertex) linearNetwork.getVertexByLabel(Y_OBSERVATION_LABEL).removeLabel();
        DoubleVertex yVertex = probabilities.sigmoid();
        BernoulliVertex outcomes = new BernoulliVertex(yVertex);
        outcomes.setLabel(Y_OBSERVATION_LABEL);
        return new BayesianNetwork(outcomes.getConnectedGraph());
    }

    private double[] regularizeSigma(DoubleTensor x) {
        int numFeatures = x.getShape()[0];
        double[] regularizedSigma = new double[numFeatures];
        for (int i = 0; i < numFeatures; i++) {
            DoubleTensor xColumn = x.slice(0, i);
            regularizedSigma[i] = priorSigma / xColumn.abs().average() / regularization;
        }
        return regularizedSigma;
    }

    @Override
    public LogisticRegression fit() {
        regression.fit(y.greaterThan(0.5));
        return this;
    }

    @Override
    public DoubleTensor predict(DoubleTensor x) {
        return regression.predict(x).sigmoid();
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
        return ((DoubleVertex) net.getVertexByLabel(WEIGHTS_LABEL)).getValue(0, index);
    }

}