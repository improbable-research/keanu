package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.continuous.MultivariateGaussian;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.testcategory.Slow;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.LogProbGraphContract;
import io.improbable.keanu.vertices.LogProbGraphValueFeeder;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.junit.rules.ExpectedException;

import static io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDoubleTensorContract.sampleMethodMatchesLogProbMethodMultiVariate;
import static io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDoubleTensorContract.sampleUnivariateMethodMatchesLogProbMethod;
import static org.junit.Assert.assertEquals;


public class MultivariateGaussianTest {

    @Rule
    public ExpectedException thrown = ExpectedException.none();

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

        sampleUnivariateMethodMatchesLogProbMethod(mvg, from, to, bucketSize, 1e-2, random, 300000);
    }

    @Test
    public void throwsIfMuIsNotRank2() {
        DoubleVertex mu = new ConstantDoubleVertex(new double[] {0.}, new long[] {1});
        DoubleVertex covariance = new ConstantDoubleVertex(new double[] {1.}, new long[] {1, 1});

        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("Ranks of mu and covariance must be 2. Given: 1, 2");

        new MultivariateGaussianVertex(mu, covariance);
    }

    @Test
    public void throwsIfCovarianceIsNotRank2() {
        DoubleVertex mu = new ConstantDoubleVertex(new double[] {0.}, new long[] {1, 1});
        DoubleVertex covariance = new ConstantDoubleVertex(new double[] {1.}, new long[] {1});

        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("Ranks of mu and covariance must be 2. Given: 2, 1");

        new MultivariateGaussianVertex(mu, covariance);
    }

    @Test
    public void throwsIfCovarianceFirstDimensionNotEqualToSecond() {
        DoubleVertex mu = new ConstantDoubleVertex(new double[] {0.}, new long[] {1, 1});
        DoubleVertex covariance = new ConstantDoubleVertex(new double[] {1., 1.}, new long[] {2, 1});

        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("Dimensions 0 and 1 of covariance must equal. Given: [2, 1]");

        new MultivariateGaussianVertex(mu, covariance);
    }

    @Test
    public void throwsIfFirstDimensionOfMuIsNot1() {
        DoubleVertex mu = new ConstantDoubleVertex(new double[] {0., 0.}, new long[] {1, 2});
        DoubleVertex covariance = new ConstantDoubleVertex(new double[] {1.}, new long[] {1, 1});

        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("Dimension 1 of mu must equal 1. Given: 2");

        new MultivariateGaussianVertex(mu, covariance);
    }

    @Test
    public void throwsIfFirstDimensionOfMuIsNotEqualToFirstDimensionOfCovariance() {
        DoubleVertex mu = new ConstantDoubleVertex(new double[] {0., 0.}, new long[] {2, 1});
        DoubleVertex covariance = new ConstantDoubleVertex(new double[] {1.}, new long[] {1, 1});

        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("Dimension 0 of mu must equal dimension 0 of covariance. Given: 2, 1");

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
        DoubleVertex mu = ConstantVertex.of(new double[]{2, 3}, 2, 1);

        MultivariateGaussianVertex mvg = new MultivariateGaussianVertex(mu, 1);

        double expectedDensity1 = new NormalDistribution(2, 1).logDensity(8);
        double expectedDensity2 = new NormalDistribution(3, 1).logDensity(10);
        double expectedDensity = expectedDensity1 + expectedDensity2;

        double density = mvg.logPdf(DoubleTensor.create(new double[]{8, 10}, 2, 1));

        assertEquals(expectedDensity, density, 0.0001);
    }

    @Test
    public void bivariateGaussianLogProbGraphMatchesLogDensityOfVector() {
        DoubleVertex mu = ConstantVertex.of(new double[]{2, 3}, 2, 1);
        MultivariateGaussianVertex mvg = new MultivariateGaussianVertex(mu, 1.);
        DoubleVertex covariance = mvg.getCovariance();
        LogProbGraph logProbGraph = mvg.logProbGraph();

        LogProbGraphValueFeeder.feedValue(logProbGraph, mu, mu.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, covariance, covariance.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, mvg, DoubleTensor.create(new double[] {8., 10.}, 2, 1));

        double expectedDensity1 = new NormalDistribution(2, 1).logDensity(8);
        double expectedDensity2 = new NormalDistribution(3, 1).logDensity(10);
        double expectedDensity = expectedDensity1 + expectedDensity2;

        LogProbGraphContract.matchesKnownLogDensity(logProbGraph, expectedDensity);
    }

    @Test
    public void bivariateGaussianLogProbMatchesLogDensityOfScipy() {
        DoubleVertex mu = ConstantVertex.of(new double[]{1, 2}, 2, 1);
        DoubleVertex covarianceMatrix = ConstantVertex.of(new double[]{1, 0.3, 0.3, 0.6}, 2, 2);

        MultivariateGaussianVertex mvg = new MultivariateGaussianVertex(mu, covarianceMatrix);
        double density = mvg.logPdf(DoubleTensor.create(new double[]{0.5, 0.4}, 2, 1));
        double expected = -3.6874792995813834;

        assertEquals(expected, density, 0.001);
    }

    @Test
    public void bivariateGaussianLogProbGraphMatchesLogDensityOfScipy() {
        DoubleVertex mu = ConstantVertex.of(new double[]{1, 2}, 2, 1);
        DoubleVertex covarianceMatrix = ConstantVertex.of(new double[]{1, 0.3, 0.3, 0.6}, 2, 2);
        MultivariateGaussianVertex mvg = new MultivariateGaussianVertex(mu, covarianceMatrix);
        LogProbGraph logProbGraph = mvg.logProbGraph();

        LogProbGraphValueFeeder.feedValue(logProbGraph, mu, mu.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, covarianceMatrix, covarianceMatrix.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, mvg, DoubleTensor.create(new double[] {0.5, 0.4}, 2, 1));

        double expected = -3.6874792995813834;
        LogProbGraphContract.matchesKnownLogDensity(logProbGraph, expected);
    }

    @Test
    public void multivariateGaussianLogProbMatchesLogDensityOfScipy() {
        DoubleVertex mu = ConstantVertex.of(new double[]{1, 2, 3}, 3, 1);

        DoubleVertex covarianceMatrix = ConstantVertex.of(
            new double[]{
                1.0, 0.3, 0.3,
                0.3, 0.8, 0.3,
                0.3, 0.3, 0.6
            },
            3, 3);

        MultivariateGaussianVertex mvg = new MultivariateGaussianVertex(mu, covarianceMatrix);
        double density = mvg.logPdf(DoubleTensor.create(new double[]{0.2, 0.3, 0.4}, 3, 1));
        double expected = -8.155504532016181;

        assertEquals(expected, density, 0.001);
    }

    @Test
    public void multivariateGaussianLogProbGraphMatchesLogDensityOfScipy() {
        DoubleVertex mu = ConstantVertex.of(new double[]{1, 2, 3}, 3, 1);

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
        LogProbGraphValueFeeder.feedValue(logProbGraph, mvg, DoubleTensor.create(new double[] {0.2, 0.3, 0.4}, 3, 1));

        double expected = -8.155504532016181;

        LogProbGraphContract.matchesKnownLogDensity(logProbGraph, expected);
    }

    @Category(Slow.class)
    @Test
    public void gaussianSampleMethodMatchesLogProbMethod() {
        DoubleVertex mu = ConstantVertex.of(new double[]{0, 0}, 2, 1);

        MultivariateGaussianVertex mvg = new MultivariateGaussianVertex(mu, 1);

        double from = -1.;
        double to = 1.;
        double bucketSize = 0.05;

        sampleMethodMatchesLogProbMethodMultiVariate(mvg, from, to, bucketSize, 0.01, 100000, random, bucketSize * bucketSize, false);
    }

    @Test(expected = IllegalArgumentException.class)
    public void whenYouSampleYouMustMatchMusShape() {
        DoubleTensor mu = DoubleTensor.create(new double[]{0, 0}, 2, 1);
        DoubleTensor sigma = DoubleTensor.create(new double[]{1}, 1);

        ContinuousDistribution mvg = MultivariateGaussian.withParameters(mu, sigma);
        mvg.sample(new long[]{2, 2}, KeanuRandom.getDefaultRandom());
    }
}
