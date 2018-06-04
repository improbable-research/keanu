package io.improbable.keanu.e2e.regression;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.algorithms.variational.GradientOptimizer;
import io.improbable.keanu.algorithms.variational.tensor.TensorGradientOptimizer;
import io.improbable.keanu.network.BayesNetDoubleAsContinuous;
import io.improbable.keanu.network.BayesNetTensorAsContinuous;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDouble;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.ConstantDoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.probabilistic.TensorGaussianVertex;
import io.improbable.keanu.vertices.dbltensor.probabilistic.TensorUniformVertex;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class LinearRegression {
    private final Logger log = LoggerFactory.getLogger(LinearRegression.class);

    @Rule
    public DeterministicRule deterministicRule = new DeterministicRule();

    private KeanuRandom random;

    private class Point {
        public final double y;
        public final double[] factors;

        public Point(double y, double factors[]) {
            this.y = y;
            this.factors = factors;
        }
    }

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void linearRegression1FactorVariationalMAP() {

        double expectedM = 3.0;
        double expectedB = 20.0;
        List<Point> points = make1FactorTestData(10000, expectedM, expectedB);

        ProbabilisticDouble m = new GaussianVertex(0.0, 10.0);
        ProbabilisticDouble b = new GaussianVertex(0.0, 10.0);

        log.info("Building graph");
        for (Point p : points) {
            DoubleVertex d = m.multiply(p.factors[0]).plus(b);
            ProbabilisticDouble dProb = new GaussianVertex(d, 5.0);
            dProb.observe(p.y);
        }

        m.setValue(1.0);
        b.setAndCascade(-5.0);

        BayesNetDoubleAsContinuous bayesNet = new BayesNetDoubleAsContinuous(m.getConnectedGraph());
        runGradientOptimizer(bayesNet, 10000, 10000);

        log.info("M = " + m.getValue() + ", B = " + b.getValue());
        assertEquals(expectedM, m.getValue(), 0.01);
        assertEquals(expectedB, b.getValue(), 0.01);
    }


    @Test
    public void linearRegression1FactorTensorVariationalMAP() {

        // Generate data
        int N = 100000;
        double expectedM = 3.0;
        double expectedB = 20.0;

        DoubleTensorVertex xGenerator = new TensorUniformVertex(new int[]{1, N}, 0, 10);
        DoubleTensorVertex yGenerator = new TensorGaussianVertex(xGenerator.multiply(expectedM).plus(expectedB), 1.0);
        DoubleTensor xData = xGenerator.sample(random);
        xGenerator.setAndCascade(xData);
        DoubleTensor yData = yGenerator.sample(random);

        // Linear Regression
        DoubleTensorVertex m = new TensorGaussianVertex(0.0, 10.0);
        DoubleTensorVertex b = new TensorGaussianVertex(0.0, 10.0);
        DoubleTensorVertex x = new ConstantDoubleTensorVertex(xData);
        DoubleTensorVertex y = new TensorGaussianVertex(x.multiply(m).plus(b), 5.0);
        y.observe(yData);

        BayesNetTensorAsContinuous bayesNet = new BayesNetTensorAsContinuous(m.getConnectedGraph());
        TensorGradientOptimizer optimizer = new TensorGradientOptimizer(bayesNet);

        optimizer.maxLikelihood(10000);

        log.info("M = " + m.getValue().scalar() + ", B = " + b.getValue().scalar());
        assertEquals(expectedM, m.getValue().scalar(), 0.05);
        assertEquals(expectedB, b.getValue().scalar(), 0.05);
    }

    @Test
    public void linearRegression2FactorVariationalMAP() {

        double expectedA = 1.0;
        double expectedB = 2.0;
        double expectedC = 3.0;

        ProbabilisticDouble a = new UniformVertex(-10.0, 10.0);
        ProbabilisticDouble b = new UniformVertex(-10.0, 10.0);
        ProbabilisticDouble c = new UniformVertex(-10.0, 10.0);

        List<Point> points = make2FactorTestData(1000, expectedA, expectedB, expectedC);

        for (Point p : points) {
            DoubleVertex d = a.multiply(p.factors[0]).plus(b.multiply(p.factors[1])).plus(c);
            ProbabilisticDouble dProb = new GaussianVertex(d, 5.0);
            dProb.observe(p.y);
        }

        runGradientOptimizer(new BayesNetDoubleAsContinuous(a.getConnectedGraph()), 1000, 10000);

        log.info("A = " + a.getValue() + ", B = " + b.getValue() + ", C = " + c.getValue());
        assertEquals(expectedA, a.getValue(), 0.01);
        assertEquals(expectedB, b.getValue(), 0.01);
        assertEquals(expectedC, c.getValue(), 0.01);
    }

    private void runGradientOptimizer(BayesNetDoubleAsContinuous bayesNet, int findStartStateAttempts, int maxEvaluations) {
        log.info("Preparing graph");
        bayesNet.probeForNonZeroMasterP(findStartStateAttempts, random);
        log.info("Running optimizer");
        GradientOptimizer optimizer = new GradientOptimizer(bayesNet);
        optimizer.maxLikelihood(maxEvaluations);
    }

    private List<Point> make1FactorTestData(int N, double m, double b) {

        List<Point> points = new ArrayList<>();

        for (int x = 0; x < N; x++) {
            double y = (m * x) + b;
            double[] factors = {x};
            points.add(new Point(y, factors));
        }

        return points;
    }

    private List<Point> make2FactorTestData(int N, double a, double b, double c) {

        List<Point> points = new ArrayList<>();

        int Ndim = (int) Math.sqrt(N);

        for (int i = 0; i < Ndim; i++) {
            for (int j = 0; j < Ndim; j++) {
                double y = (a * i) + (b * j) + c;
                double[] factors = {i, j};
                points.add(new Point(y, factors));
            }
        }

        return points;
    }
}