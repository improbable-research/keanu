package io.improbable.keanu.model.linear;

import com.google.common.primitives.Ints;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

public class LogisticRegression implements ClassificationModel {

    private static final double DEFAULT_MU = 0.0;
    private static final double DEFAULT_SIGMA = 2.0;
    private static final double DEFAULT_REGULARIZATION = 1.0;

    private final BayesianNetwork net;
    private final BooleanTensor y;

    public LogisticRegression(DoubleTensor x, BooleanTensor y, double regularization, double priorMu, double priorSigma) {
        this.y = y;
        this.net = buildModel(x, priorMu, priorSigma, regularizeSigma(x, priorSigma, regularization));
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

    private BayesianNetwork buildModel(DoubleTensor x, double priorMu, double priorSigma, double... priorOnWeights) {
        BayesianNetwork net = LinearRegressionGraph.build(x, priorMu, priorSigma, priorOnWeights);
        DoubleVertex probabilities = (DoubleVertex) net.getVertexByLabel(Y_LABEL).removeLabel();
        DoubleVertex sigmoid = probabilities.sigmoid().setLabel(Y_LABEL);
        BernoulliVertex outcome = new BernoulliVertex(sigmoid);
        outcome.setLabel(Y_OBSERVATION_LABEL);
        return new BayesianNetwork(outcome.getConnectedGraph());
    }

    private double[] regularizeSigma(DoubleTensor x, double priorSigma, double regularization) {
        int numFeatures = Ints.saturatedCast(x.getShape()[0]);
        double[] regularizedSigma = new double[numFeatures];
        for (int i = 0; i < numFeatures; i++) {
            DoubleTensor xColumn = x.slice(0, i);
            regularizedSigma[i] = priorSigma / xColumn.abs().average() / regularization;
        }
        return regularizedSigma;
    }

}