package io.improbable.keanu.model;

import io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizer;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

public class LogisticRegression implements LinearModel {

    private static final double DEFAULT_MU = 0.0;
    private static final double DEFAULT_SIGMA = 2.0;
    private static final double DEFAULT_REGULARIZATION = 1;
    private static final VertexLabel WEIGHTS_LABEL = new VertexLabel("weights");
    private static final VertexLabel INTERCEPT_LABEL = new VertexLabel("intercept");
    private static final VertexLabel PROBABILITIES_LABEL = new VertexLabel("probabilities");
    private static final VertexLabel OUTCOMES_LABEL = new VertexLabel("outcomes");

    private DoubleTensor x;
    private DoubleTensor y;
    private BayesianNetwork net;
    private double priorMu;
    private double priorSigma;
    private boolean isFit = false;
    private double regularization;


    public LogisticRegression(DoubleTensor x, DoubleTensor y, double regularization) {
        this.x = x;
        this.y = y;
        this.priorMu = DEFAULT_MU;
        this.priorSigma = DEFAULT_SIGMA;
        this.regularization = regularization;
        construct();
    }

    public LogisticRegression(DoubleTensor x, DoubleTensor y) {
        this(x, y, DEFAULT_REGULARIZATION);
    }

    @Override
    public BayesianNetwork construct() {
        DoubleVertex probabilities = computeProbabilities(x);
        probabilities.setLabel(PROBABILITIES_LABEL);
        BernoulliVertex outcomes = new BernoulliVertex(probabilities);
        outcomes.setLabel(OUTCOMES_LABEL);
        net = new BayesianNetwork(outcomes.getConnectedGraph());
        return net;
    }

    private DoubleVertex computeProbabilities(DoubleTensor x) {
        int numFeatures = x.getShape()[0];
        double[] sigma = new double[numFeatures];
        for (int i = 0; i < numFeatures; i++) {
            DoubleTensor xColumn = x.slice(0, i);
            sigma[i] = priorSigma / xColumn.abs().average() / regularization;
        }
        DoubleVertex intercept = new GaussianVertex(priorMu, priorSigma);
        intercept.setLabel(INTERCEPT_LABEL);
        DoubleVertex weights = new GaussianVertex(priorMu, ConstantVertex.of(sigma));
        weights.setLabel(WEIGHTS_LABEL);

        ConstantDoubleVertex xVertex = ConstantVertex.of(x);
        return weights.getValue().isScalar() ?
            weights.times(xVertex).plus(intercept).sigmoid() :
            weights.matrixMultiply(xVertex).plus(intercept).sigmoid();
    }

    @Override
    public LogisticRegression fit() {
        BooleanTensor observedOutcomes = y.greaterThan(0.5);
        net.getVertexByLabel(OUTCOMES_LABEL).observe(observedOutcomes);
        GradientOptimizer optimizer = GradientOptimizer.of(net);
        optimizer.maxAPosteriori();
        isFit = true;
        return this;
    }

    @Override
    public DoubleTensor predict(DoubleTensor x) {
        if (isFit) {
            DoubleVertex weights = (DoubleVertex) net.getVertexByLabel(WEIGHTS_LABEL);
            DoubleVertex intercept = (DoubleVertex) net.getVertexByLabel(INTERCEPT_LABEL);
            return weights.matrixMultiply(ConstantVertex.of(x)).plus(intercept).getValue().sigmoid();
        } else {
            throw new RuntimeException("The model must be fit before attempting to predict.");
        }
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