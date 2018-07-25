package io.improbable.keanu.e2e.regression;

import static org.junit.Assert.assertEquals;

import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.algorithms.variational.GradientOptimizer;
import io.improbable.keanu.distributions.dual.ParameterName;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.probabilistic.DistributionVertexBuilder;
import io.improbable.keanu.vertices.dbl.probabilistic.VertexOfType;

public class LinearRegression {
    private final Logger log = LoggerFactory.getLogger(LinearRegression.class);

    @Rule
    public DeterministicRule deterministicRule = new DeterministicRule();

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void linearRegression1FactorTensorVariationalMAP() {

        // Generate data
        int N = 100000;
        double expectedM = 3.0;
        double expectedB = 20.0;

        DoubleVertex xGenerator = new DistributionVertexBuilder()
            .shaped(1, N)
            .withInput(ParameterName.MIN, 0.)
            .withInput(ParameterName.MAX, 10.)
            .uniform();
        DoubleVertex mu = xGenerator.multiply(expectedM).plus(expectedB);
        DoubleVertex yGenerator = VertexOfType.gaussian(mu, ConstantVertex.of(1.0));
        DoubleTensor xData = xGenerator.sample(random);
        xGenerator.setAndCascade(xData);
        DoubleTensor yData = yGenerator.sample(random);

        // Linear Regression
        DoubleVertex m = VertexOfType.gaussian(0.0, 10.0);
        DoubleVertex b = VertexOfType.gaussian(0.0, 10.0);
        DoubleVertex x = ConstantVertex.of(xData);
        DoubleVertex y = VertexOfType.gaussian(x.multiply(m).plus(b), ConstantVertex.of(5.0));
        y.observe(yData);

        BayesianNetwork bayesNet = new BayesianNetwork(m.getConnectedGraph());
        GradientOptimizer optimizer = new GradientOptimizer(bayesNet);

        optimizer.maxLikelihood();

        log.info("M = " + m.getValue().scalar() + ", B = " + b.getValue().scalar());
        assertEquals(expectedM, m.getValue().scalar(), 0.05);
        assertEquals(expectedB, b.getValue().scalar(), 0.05);
    }

    @Test
    public void linearRegressionTwoFactorTensorVariationalMAP() {

        // Generate data
        int N = 100000;
        double expectedW1 = 3.0;
        double expectedW2 = 7.0;
        double expectedB = 20.0;

        DoubleVertex x1Generator = new DistributionVertexBuilder()
            .shaped(1, N)
            .withInput(ParameterName.MIN, 0.)
            .withInput(ParameterName.MAX, 10.)
            .uniform();
        DoubleVertex x2Generator = new DistributionVertexBuilder()
            .shaped(1, N)
            .withInput(ParameterName.MIN, 50.)
            .withInput(ParameterName.MAX, 100.)
            .uniform();
        DoubleVertex yGenerator = VertexOfType.gaussian(
            x1Generator.multiply(expectedW1).plus(x2Generator.multiply(expectedW2)).plus(expectedB),
            ConstantVertex.of(1.0)
        );
        DoubleTensor x1Data = x1Generator.sample(random);
        x1Generator.setAndCascade(x1Data);
        DoubleTensor x2Data = x1Generator.sample(random);
        x2Generator.setAndCascade(x2Data);
        DoubleTensor yData = yGenerator.sample(random);

        // Linear Regression
        DoubleVertex w1 = VertexOfType.gaussian(0.0, 10.0);
        DoubleVertex w2 = VertexOfType.gaussian(0.0, 10.0);
        DoubleVertex b = VertexOfType.gaussian(0.0, 10.0);
        DoubleVertex x1 = ConstantVertex.of(x1Data);
        DoubleVertex x2 = ConstantVertex.of(x2Data);
        DoubleVertex y = VertexOfType.gaussian(x1.multiply(w1).plus(x2.multiply(w2)).plus(b), ConstantVertex.of(5.0));
        y.observe(yData);

        BayesianNetwork bayesNet = new BayesianNetwork(y.getConnectedGraph());
        GradientOptimizer optimizer = new GradientOptimizer(bayesNet);

        optimizer.maxLikelihood();

        log.info("W1 = " + w1.getValue().scalar() + " W2 = " + w2.getValue().scalar() + ", B = " + b.getValue().scalar());
        assertEquals(expectedW1, w1.getValue().scalar(), 0.05);
        assertEquals(expectedW2, w2.getValue().scalar(), 0.05);
        assertEquals(expectedB, b.getValue().scalar(), 0.05);
    }

    @Test
    public void linearRegressionManyFactorTensorVariationalMAP() {

        // Generate data
        int N = 1000;
        int featureCount = 40;

        double[] expectedWeights = new double[featureCount];
        double expectedB = 20.0;
        DoubleVertex[] xGenerators = new DoubleVertex[featureCount];
        DoubleTensor[] xData = new DoubleTensor[featureCount];
        DoubleVertex yGeneratorMu = ConstantVertex.of(0.0);
        for (int i = 0; i < expectedWeights.length; i++) {
            expectedWeights[i] = random.nextDouble() * 100 + 20;
            xGenerators[i] = new DistributionVertexBuilder()
                .shaped(1, N)
                .withInput(ParameterName.MIN, 0.)
                .withInput(ParameterName.MAX, 10000.)
                .uniform();
            xData[i] = xGenerators[i].sample(random);
            xGenerators[i].setValue(xData[i]);
            yGeneratorMu = yGeneratorMu.plus(xGenerators[i].multiply(expectedWeights[i]));
        }

        yGeneratorMu = yGeneratorMu.plus(expectedB);
        DoubleVertex yGenerator = VertexOfType.gaussian(yGeneratorMu, ConstantVertex.of(1.0));
        DoubleTensor yData = yGenerator.sample(random);

        // Run Linear Regression
        DoubleVertex[] weights = new DoubleVertex[featureCount];
        DoubleVertex[] x = new DoubleVertex[featureCount];
        DoubleVertex yMu = ConstantVertex.of(0.0);
        for (int i = 0; i < weights.length; i++) {
            weights[i] = VertexOfType.gaussian(0.0, 1.0);
            weights[i].setValue(0);
            x[i] = ConstantVertex.of(xData[i]);
            yMu = yMu.plus(x[i].multiply(weights[i]));
        }

        DoubleVertex b = VertexOfType.uniform(-50., 50.);
        DoubleVertex y = VertexOfType.gaussian(yMu.plus(b), ConstantVertex.of(1.0));
        y.observe(yData);

        BayesianNetwork bayesNet = new BayesianNetwork(y.getConnectedGraph());
        GradientOptimizer optimizer = new GradientOptimizer(bayesNet);
        optimizer.maxLikelihood();

        for (int i = 0; i < featureCount; i++) {
            assertEquals(expectedWeights[i], weights[i].getValue().scalar(), 0.05);
        }
    }

    @Test
    public void linearRegressionTwoFactorTensorWithMatrixMultiplyVariationalMAP() {

        // Generate data
        int N = 100000;
        double expectedW1 = 12.0;
        double expectedW2 = 7.0;

        DoubleVertex wGenerator = ConstantVertex.of(DoubleTensor.create(new double[]{expectedW1, expectedW2}, 2, 1));
        DoubleVertex xGenerator = new DistributionVertexBuilder()
            .shaped(N,2)
            .withInput(ParameterName.MIN, 0.)
            .withInput(ParameterName.MAX, 10.)
            .uniform();
        DoubleVertex yGenerator = VertexOfType.gaussian(
            xGenerator.matrixMultiply(wGenerator),
            ConstantVertex.of(1.0)
        );
        DoubleTensor xData = xGenerator.sample(random);
        xGenerator.setAndCascade(xData);
        DoubleTensor yData = yGenerator.sample(random);

        // Linear Regression
        DoubleVertex w =         new DistributionVertexBuilder()
            .shaped(2,1)
            .withInput(ParameterName.MU, 0.)
            .withInput(ParameterName.SIGMA, 10.)
            .gaussian();
        w.setValue(DoubleTensor.create(new double[]{2, 2}, 2, 1));

        DoubleVertex x = ConstantVertex.of(xData);
        DoubleVertex yMu = x.matrixMultiply(w);
        DoubleVertex y = VertexOfType.gaussian(yMu, ConstantVertex.of(5.0));
        y.observe(yData);

        BayesianNetwork bayesNet = new BayesianNetwork(y.getConnectedGraph());
        GradientOptimizer optimizer = new GradientOptimizer(bayesNet);

        optimizer.maxLikelihood();

        System.out.println("W1 = " + w.getValue(0) + " W2 = " + w.getValue(1));
        assertEquals(expectedW1, w.getValue(0), 0.05);
        assertEquals(expectedW2, w.getValue(1), 0.05);
    }

}