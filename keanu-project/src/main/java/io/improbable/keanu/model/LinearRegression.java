package io.improbable.keanu.model;

import io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizer;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

public class LinearRegression implements LinearModel {

    private static final double DEFAULT_PRIOR_ON_WEIGHTS = 2.0;
    private static final double MU_OF_WEIGHTS = 0.0;
    private static final VertexLabel WEIGHTS_LABEL = new VertexLabel("weights");
    private static final VertexLabel INTERCEPT_LABEL = new VertexLabel("intercept");
    private static final VertexLabel Y_OBSERVATION_LABEL = new VertexLabel("y");

    private DoubleTensor x;
    private DoubleTensor y;
    private BayesianNetwork net;
    private boolean isFit;
    private double sigmaOnPrior;

    public LinearRegression(DoubleTensor x, DoubleTensor y) {
        this(x, y, DEFAULT_PRIOR_ON_WEIGHTS);
    }

    public LinearRegression(DoubleTensor x, DoubleTensor y, double sigmaOnPrior) {
        this.x = x;
        this.y = y;
        this.isFit = false;
        this.net = null;
        this.sigmaOnPrior = sigmaOnPrior;
        construct();
    }

    @Override
    public BayesianNetwork construct() {
        int numberOfFeatures = x.getShape()[0];
        int[] shapeOfWeights = new int[]{1, numberOfFeatures};
        DoubleVertex weights = new GaussianVertex(shapeOfWeights, MU_OF_WEIGHTS, sigmaOnPrior).setLabel(WEIGHTS_LABEL);
        DoubleVertex intercept = new GaussianVertex(MU_OF_WEIGHTS, sigmaOnPrior).setLabel(INTERCEPT_LABEL);
        DoubleVertex xMu = weights.getValue().isScalar() ? weights.times(ConstantVertex.of(x)) : weights.matrixMultiply(ConstantVertex.of(x));
        DoubleVertex yVertex = new GaussianVertex(xMu.plus(intercept), sigmaOnPrior).setLabel(Y_OBSERVATION_LABEL);
        net = new BayesianNetwork(yVertex.getConnectedGraph());
        return net;
    }

    @Override
    public LinearRegression fit() {
        net.getVertexByLabel(Y_OBSERVATION_LABEL).observe(y);
        GradientOptimizer optimizer = GradientOptimizer.of(net);
        optimizer.maxLikelihood();
        isFit = true;
        return this;
    }

    @Override
    public DoubleTensor predict(DoubleTensor x) {
        if (isFit) {
            DoubleVertex weights = (DoubleVertex) net.getVertexByLabel(WEIGHTS_LABEL);
            DoubleVertex intercept = (DoubleVertex) net.getVertexByLabel(INTERCEPT_LABEL);
            return weights.getValue().times(x).plus(intercept.getValue());
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
