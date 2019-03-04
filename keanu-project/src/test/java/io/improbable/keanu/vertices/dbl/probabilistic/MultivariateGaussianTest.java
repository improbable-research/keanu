package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.continuous.MultivariateGaussian;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.testcategory.Slow;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.LogProbGraphContract;
import io.improbable.keanu.vertices.LogProbGraphValueFeeder;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.Differentiator;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.LogProbGradientCalculator;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialsOf;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.junit.rules.ExpectedException;

import java.util.HashSet;
import java.util.Map;

import static io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDoubleTensorContract.sampleMethodMatchesLogProbMethodMultiVariate;
import static io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDoubleTensorContract.sampleUnivariateMethodMatchesLogProbMethod;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;


public class MultivariateGaussianTest {

    @Rule
    public ExpectedException thrown = ExpectedException.none();

    @Rule
    public DeterministicRule deterministicRule = new DeterministicRule();

    KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Category(Slow.class)
    @Test
    public void samplingFromUnivariateGaussianMatchesLogDensity() {
        MultivariateGaussianVertex mvg = new MultivariateGaussianVertex(0, 1);

        double from = -2.;
        double to = 2.;
        double bucketSize = 0.1;

        sampleUnivariateMethodMatchesLogProbMethod(mvg, from, to, bucketSize, 1e-2, random, 250000);
    }

    @Test
    public void throwsIfCovarianceIsNotRank2() {
        DoubleVertex mu = new ConstantDoubleVertex(new double[]{0.}, new long[]{1});
        DoubleVertex covariance = new ConstantDoubleVertex(new double[]{1.}, new long[]{1});

        thrown.expect(IllegalArgumentException.class);

        new MultivariateGaussianVertex(mu, covariance);
    }

    @Test
    public void throwsIfCovarianceFirstDimensionNotEqualToSecond() {
        DoubleVertex mu = new ConstantDoubleVertex(new double[]{0.}, new long[]{1, 1});
        DoubleVertex covariance = new ConstantDoubleVertex(new double[]{1., 1.}, new long[]{2, 1});

        thrown.expect(IllegalArgumentException.class);

        new MultivariateGaussianVertex(mu, covariance);
    }

    @Test
    public void throwsIfFirstDimensionOfMuIsNotEqualToFirstDimensionOfCovariance() {
        DoubleVertex mu = new ConstantDoubleVertex(new double[]{0., 0.}, new long[]{2});
        DoubleVertex covariance = new ConstantDoubleVertex(new double[]{1.}, new long[]{1, 1});

        thrown.expect(IllegalArgumentException.class);

        new MultivariateGaussianVertex(mu, covariance);
    }

    @Test
    public void logProbMatchesLogDensityOfScalar() {
        MultivariateGaussianVertex mvg = new MultivariateGaussianVertex(5, 1);

        double expectedDensity = new NormalDistribution(5.0, 1).logDensity(0.5);
        double density = mvg.logPdf(DoubleTensor.scalar(0.5));

        assertEquals(expectedDensity, density, 1e-2);
    }

    @Test
    public void logProbGraphMatchesLogDensityOfScalar() {
        MultivariateGaussianVertex mvg = new MultivariateGaussianVertex(5., 1.);
        DoubleVertex mu = mvg.getMu();
        DoubleVertex covariance = mvg.getCovariance();
        LogProbGraph logProbGraph = mvg.logProbGraph();

        LogProbGraphValueFeeder.feedValue(logProbGraph, mu, mu.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, covariance, covariance.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, mvg, DoubleTensor.scalar(0.5));

        double expectedDensity = new NormalDistribution(5.0, 1).logDensity(0.5);
        LogProbGraphContract.matchesKnownLogDensity(logProbGraph, expectedDensity);
    }

    @Test
    public void bivariateGaussianLogProbMatchesLogDensityOfVector() {
        DoubleVertex mu = ConstantVertex.of(new double[]{2, 3});

        MultivariateGaussianVertex mvg = new MultivariateGaussianVertex(mu, 1);

        double expectedDensity1 = new NormalDistribution(2, 1).logDensity(8);
        double expectedDensity2 = new NormalDistribution(3, 1).logDensity(10);
        double expectedDensity = expectedDensity1 + expectedDensity2;

        double density = mvg.logPdf(DoubleTensor.create(new double[]{8, 10}, 2));

        assertEquals(expectedDensity, density, 0.0001);
    }

    @Test
    public void bivariateGaussianLogProbGraphMatchesLogDensityOfVector() {
        DoubleVertex mu = ConstantVertex.of(new double[]{2, 3});
        MultivariateGaussianVertex mvg = new MultivariateGaussianVertex(mu, 1.);
        DoubleVertex covariance = mvg.getCovariance();
        LogProbGraph logProbGraph = mvg.logProbGraph();

        LogProbGraphValueFeeder.feedValue(logProbGraph, mu, mu.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, covariance, covariance.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, mvg, DoubleTensor.create(new double[]{8., 10.}, 2));

        double expectedDensity1 = new NormalDistribution(2, 1).logDensity(8);
        double expectedDensity2 = new NormalDistribution(3, 1).logDensity(10);
        double expectedDensity = expectedDensity1 + expectedDensity2;

        LogProbGraphContract.matchesKnownLogDensity(logProbGraph, expectedDensity);
    }

    @Test
    public void bivariateGaussianLogProbMatchesLogDensityOfScipy() {
        DoubleVertex mu = ConstantVertex.of(new double[]{1, 2});
        DoubleVertex covarianceMatrix = ConstantVertex.of(new double[]{1, 0.3, 0.3, 0.6}, 2, 2);

        MultivariateGaussianVertex mvg = new MultivariateGaussianVertex(mu, covarianceMatrix);
        double density = mvg.logPdf(DoubleTensor.create(new double[]{0.5, 0.4}, 2));
        double expected = -3.6874792995813834;

        assertEquals(expected, density, 0.001);
    }

    @Test
    public void bivariateGaussianLogProbGraphMatchesLogDensityOfScipy() {
        DoubleVertex mu = ConstantVertex.of(new double[]{1, 2});
        DoubleVertex covarianceMatrix = ConstantVertex.of(new double[]{1, 0.3, 0.3, 0.6}, 2, 2);
        MultivariateGaussianVertex mvg = new MultivariateGaussianVertex(mu, covarianceMatrix);
        LogProbGraph logProbGraph = mvg.logProbGraph();

        LogProbGraphValueFeeder.feedValue(logProbGraph, mu, mu.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, covarianceMatrix, covarianceMatrix.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, mvg, DoubleTensor.create(new double[]{0.5, 0.4}, 2));

        double expected = -3.6874792995813834;
        LogProbGraphContract.matchesKnownLogDensity(logProbGraph, expected);
    }

    @Test
    public void multivariateGaussianLogProbMatchesLogDensityOfScipy() {
        DoubleVertex mu = ConstantVertex.of(new double[]{1, 2, 3});

        DoubleVertex covarianceMatrix = ConstantVertex.of(
            new double[]{
                1.0, 0.3, 0.3,
                0.3, 0.8, 0.3,
                0.3, 0.3, 0.6
            },
            3, 3);

        MultivariateGaussianVertex mvg = new MultivariateGaussianVertex(mu, covarianceMatrix);
        double density = mvg.logPdf(DoubleTensor.create(new double[]{0.2, 0.3, 0.4}, 3));
        double expected = -8.155504532016181;

        assertEquals(expected, density, 0.001);
    }

    @Test
    public void multivariateGaussianLogProbGraphMatchesLogDensityOfScipy() {
        DoubleVertex mu = ConstantVertex.of(new double[]{1, 2, 3}, 3);

        DoubleVertex covarianceMatrix = ConstantVertex.of(
            new double[]{
                1.0, 0.3, 0.3,
                0.3, 0.8, 0.3,
                0.3, 0.3, 0.6
            },
            3, 3);

        MultivariateGaussianVertex mvg = new MultivariateGaussianVertex(mu, covarianceMatrix);
        LogProbGraph logProbGraph = mvg.logProbGraph();

        LogProbGraphValueFeeder.feedValue(logProbGraph, mu, mu.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, covarianceMatrix, covarianceMatrix.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, mvg, DoubleTensor.create(new double[]{0.2, 0.3, 0.4}, 3));

        double expected = -8.155504532016181;

        LogProbGraphContract.matchesKnownLogDensity(logProbGraph, expected);
    }

    @Category(Slow.class)
    @Test
    public void gaussianSampleMethodMatchesLogProbMethod() {
        DoubleVertex mu = ConstantVertex.of(new double[]{0, 0}, 2);

        MultivariateGaussianVertex mvg = new MultivariateGaussianVertex(mu, 1);

        double from = -1.;
        double to = 1.;
        double bucketSize = 0.05;

        sampleMethodMatchesLogProbMethodMultiVariate(mvg, from, to, bucketSize, 0.01, 10000, random, bucketSize * bucketSize, false);
    }

    @Test(expected = IllegalArgumentException.class)
    public void whenYouSampleYouMustMatchMusShape() {
        DoubleTensor mu = DoubleTensor.create(new double[]{0, 0}, 2, 1);
        DoubleTensor sigma = DoubleTensor.create(new double[]{1}, 1);

        ContinuousDistribution mvg = MultivariateGaussian.withParameters(mu, sigma);
        mvg.sample(new long[]{2, 2}, KeanuRandom.getDefaultRandom());
    }

    @Test
    public void autoDiffOfLogProbGraphEqualsDlogProb() {
        DoubleVertex mu = ConstantVertex.of(DoubleTensor.create(1, 2));
        ConstantDoubleVertex cov = ConstantVertex.of(DoubleTensor.create(1, 0, 0, 2).reshape(2, 2));
        MultivariateGaussianVertex A = new MultivariateGaussianVertex(mu, cov);
        A.setValue(DoubleTensor.create(1, 3));

        BayesianNetwork bayesianNetwork = new BayesianNetwork(A.getConnectedGraph());
        LogProbGradientCalculator logProbGradientCalculator = new LogProbGradientCalculator(
            bayesianNetwork.getContinuousLatentVertices(),
            bayesianNetwork.getContinuousLatentVertices()
        );

        LogProbGraph logProbGraph = A.logProbGraph();

        logProbGraph.getInput(A).setValue(A.getValue());
        logProbGraph.getInput(mu).setValue(mu.getValue());
        logProbGraph.getInput(cov).setValue(cov.getValue());

        PartialsOf partialsOf = Differentiator.reverseModeAutoDiff(
            logProbGraph.getLogProbOutput(),
            new HashSet<>(logProbGraph.getInputs().values())
        );

        DoubleTensor expected = partialsOf.withRespectTo(A);

        Map<VertexId, DoubleTensor> gradient = logProbGradientCalculator.getJointLogProbGradientWrtLatents();
        DoubleTensor actual = gradient.get(A.getId());

        assertArrayEquals(expected.asFlatDoubleArray(), actual.asFlatDoubleArray(), 1e-8);
    }
}
