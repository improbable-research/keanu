package io.improbable.keanu.model;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.DirichletVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.generic.probabilistic.discrete.CategoricalVertex;
import io.improbable.keanu.vertices.intgr.probabilistic.BinomialVertex;

public class LogisticRegression implements LinearModel {

    private static final double DEFAULT_MU = 0.0;
    private static final double DEFAULT_SIGMA = 2.0;
    private static final double DEFAULT_REGULARIZATION = 1;

    private final double priorSigma;
    private final double regularization;
    private final LinearRegression regression;
    private final BayesianNetwork net;

    public LogisticRegression(DoubleTensor x, BooleanTensor y, double regularization, double priorMu, double priorSigma) {
        this.priorSigma = priorSigma;
        this.regularization = regularization;
        this.regression = new LinearRegression(x, y.toDoubleMask(), priorMu, priorSigma, regularizeSigma(x));
        this.net = construct();
        this.regression.setNet(net);
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
    public BayesianNetwork construct() {
        BayesianNetwork linearNetwork = regression.getNet();
        DoubleVertex probabilities = (DoubleVertex) linearNetwork.getVertexByLabel(Y_OBSERVATION_LABEL).removeLabel();
        DoubleVertex sigmoid = probabilities.sigmoid();
        BernoulliVertex outcome = new BernoulliVertex(sigmoid);
        outcome.setLabel(Y_OBSERVATION_LABEL);
        return new BayesianNetwork(probabilities.getConnectedGraph());
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
    public boolean isFit() {
        return true;
    }

    public DoubleVertex getWeights() {
        return (DoubleVertex) net.getVertexByLabel(WEIGHTS_LABEL);
    }

    public double getWeight(int index) {
        return ((DoubleVertex) net.getVertexByLabel(WEIGHTS_LABEL)).getValue(0, index);
    }

}