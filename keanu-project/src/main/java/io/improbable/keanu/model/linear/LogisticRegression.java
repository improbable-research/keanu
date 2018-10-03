package io.improbable.keanu.model.linear;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

public class LogisticRegression implements ClassificationModel {

    private static final double DEFAULT_MU = 0.0;
    private static final double DEFAULT_SIGMA = 2.0;
    private static final double DEFAULT_REGULARIZATION = 1;

    private final double priorMu;
    private final double priorSigma;
    private final double regularization;
    private final BayesianNetwork net;
    private final DoubleTensor x;
    private final BooleanTensor y;

    public LogisticRegression(DoubleTensor x, BooleanTensor y, double regularization, double priorMu, double priorSigma) {
        this.x = x;
        this.y = y;
        this.priorMu = priorMu;
        this.priorSigma = priorSigma;
        this.regularization = regularization;
        this.net = buildModel();
    }

    public LogisticRegression(DoubleTensor x, BooleanTensor y) {
        this(x, y, DEFAULT_REGULARIZATION, DEFAULT_MU, DEFAULT_SIGMA);
    }

    public LogisticRegression(DoubleTensor x, BooleanTensor y, double regularization) {
        this(x, y, regularization, DEFAULT_MU, DEFAULT_SIGMA);
    }

    public LogisticRegression(DoubleTensor x, BooleanTensor y, double priorMu, double priorSigma) {
        this(x, y, DEFAULT_REGULARIZATION, priorMu, priorSigma);
    }

    @Override
    public BayesianNetwork buildModel() {
        BayesianNetwork net = LinearRegressionGraph.build(x, priorMu, priorSigma, regularizeSigma(x));
        DoubleVertex probabilities = (DoubleVertex) net.getVertexByLabel(Y_LABEL).removeLabel();
        DoubleVertex sigmoid = probabilities.sigmoid().setLabel(Y_LABEL);
        BernoulliVertex outcome = new BernoulliVertex(sigmoid);
        outcome.setLabel(Y_OBSERVATION_LABEL);
        return new BayesianNetwork(outcome.getConnectedGraph());
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
    public BayesianNetwork getNet() {
        return net;
    }

    @Override
    public BooleanTensor getY() {
        return y;
    }

    public DoubleVertex getWeights() {
        return (DoubleVertex) net.getVertexByLabel(WEIGHTS_LABEL);
    }

    public double getWeight(int index) {
        return ((DoubleVertex) net.getVertexByLabel(WEIGHTS_LABEL)).getValue(0, index);
    }

}