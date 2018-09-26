package io.improbable.keanu.model;

import io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizer;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

import java.util.Arrays;

public class LinearRegression implements LinearModel {

    private static final double DEFAULT_MU = 0.0;
    private static final double DEFAULT_SIGMA = 2.0;
    private static final VertexLabel WEIGHTS_LABEL = new VertexLabel("weights");
    private static final VertexLabel INTERCEPT_LABEL = new VertexLabel("intercept");
    private static final VertexLabel Y_OBSERVATION_LABEL = new VertexLabel("y");

    private final DoubleTensor x;
    private final DoubleTensor y;
    private final double priorOnSigma;
    private final double priorOnMu;
    private final double[] priorOnSigmaForWeights;

    private BayesianNetwork net;
    private boolean isFit = false;

    public LinearRegression(DoubleTensor x, DoubleTensor y) {
        this(x, y, DEFAULT_MU, DEFAULT_SIGMA);
    }

    public LinearRegression(DoubleTensor x, DoubleTensor y, double priorOnMu, double priorOnSigma) {
        this(x, y, priorOnMu, priorOnSigma, fillPriorOnWeights(x, priorOnSigma));
    }

    public LinearRegression(DoubleTensor x, DoubleTensor y, double priorOnMu, double priorOnSigma, double... priorOnSigmaForWeights) {
        this.x = x;
        this.y = y;
        this.priorOnSigma = priorOnSigma;
        this.priorOnMu = priorOnMu;
        this.priorOnSigmaForWeights = priorOnSigmaForWeights;
        this.net = construct();
    }

    private static double[] fillPriorOnWeights(DoubleTensor x, double priorOnSigma) {
        double[] priorWeights = new double[x.getShape()[0]];
        Arrays.fill(priorWeights, priorOnSigma);
        return priorWeights;
    }

    @Override
    public BayesianNetwork construct() {
        int numberOfFeatures = x.getShape()[0];
        int[] shapeOfWeights = new int[]{1, numberOfFeatures};
        DoubleVertex weights = new GaussianVertex(priorOnMu, ConstantVertex.of(priorOnSigmaForWeights)).reshape(shapeOfWeights).setLabel(WEIGHTS_LABEL);
        DoubleVertex intercept = new GaussianVertex(priorOnMu, priorOnSigma).setLabel(INTERCEPT_LABEL);
        DoubleVertex xMu = weights.getValue().isScalar() ? weights.times(ConstantVertex.of(x)) : weights.matrixMultiply(ConstantVertex.of(x));
        DoubleVertex yVertex = new GaussianVertex(xMu.plus(intercept), priorOnSigma).setLabel(Y_OBSERVATION_LABEL);
        return new BayesianNetwork(yVertex.getConnectedGraph());
    }

    @Override
    public LinearRegression fit() {
        return fit(y);
    }

    public LinearRegression fit(Tensor y) {
        net.getVertexByLabel(Y_OBSERVATION_LABEL).observe(y);
        net.probeForNonZeroProbability(100);
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
            return weights.getValue().isScalar() ?
                weights.times(ConstantVertex.of(x)).plus(intercept).getValue() :
                weights.matrixMultiply(ConstantVertex.of(x)).plus(intercept).getValue();
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

    public void setNet(BayesianNetwork net) {
        this.net = net;
    }

    public double getWeight(int index) {
        DoubleVertex weight = (DoubleVertex) net.getVertexByLabel(WEIGHTS_LABEL);
        return weight.getValue().isScalar() ? weight.getValue().scalar() : weight.getValue(0, index);
    }

}
