package io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.continuous.MultivariateGaussian;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.testcategory.Slow;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.LogProbGraphContract;
import io.improbable.keanu.vertices.LogProbGraphValueFeeder;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiator;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.LogProbGradientCalculator;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.PartialsOf;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.junit.rules.ExpectedException;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static io.improbable.keanu.tensor.TensorMatchers.valuesAndShapesMatch;
import static io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.ProbabilisticDoubleTensorContract.moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues;
import static io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.ProbabilisticDoubleTensorContract.sampleMethodMatchesLogProbMethodMultiVariate;
import static io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.ProbabilisticDoubleTensorContract.sampleUnivariateMethodMatchesLogProbMethod;
import static org.hamcrest.MatcherAssert.assertThat;
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
    public void throwsIfMuIsNotRank1() {
        DoubleVertex mu = new ConstantDoubleVertex(new double[]{0.}, new long[]{1, 1});
        DoubleVertex covariance = new ConstantDoubleVertex(new double[]{1.}, new long[]{1, 1});

        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("Mu must be vector but was rank 2");

        new MultivariateGaussianVertex(mu, covariance);
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
        double sigma = 1.5;
        MultivariateGaussianVertex mvg = new MultivariateGaussianVertex(5, sigma * sigma);

        double expectedDensity = new NormalDistribution(5.0, sigma).logDensity(0.5);
        double density = mvg.logPdf(DoubleTensor.scalar(0.5));

        assertEquals(expectedDensity, density, 1e-2);
    }

    @Test
    public void logProbGraphMatchesLogDensityOfScalar() {
        double sigma = 1.5;
        MultivariateGaussianVertex mvg = new MultivariateGaussianVertex(5., sigma * sigma);
        DoubleVertex mu = mvg.getMu();
        DoubleVertex covariance = mvg.getCovariance();
        LogProbGraph logProbGraph = mvg.logProbGraph();

        LogProbGraphValueFeeder.feedValue(logProbGraph, mu, mu.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, covariance, covariance.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, mvg, DoubleTensor.scalar(0.5));

        double expectedDensity = new NormalDistribution(5.0, sigma).logDensity(0.5);
        LogProbGraphContract.matchesKnownLogDensity(logProbGraph, expectedDensity);
    }

    @Test
    public void bivariateGaussianLogProbMatchesLogDensity() {
        DoubleVertex mu = ConstantVertex.of(new double[]{-1, 2});
        DoubleVertex covariance = ConstantVertex.of(DoubleTensor.create(0.5, 0, 0, 1).reshape(2, 2));

        MultivariateGaussianVertex mvg = new MultivariateGaussianVertex(mu, covariance);

        double expectedDensity1 = new NormalDistribution(-1, Math.sqrt(0.5)).logDensity(8);
        double expectedDensity12 = new NormalDistribution(-1, Math.sqrt(0.5)).logDensity(4);
        double expectedDensity2 = new NormalDistribution(2, Math.sqrt(1)).logDensity(10);
        double expectedDensity22 = new NormalDistribution(2, Math.sqrt(1)).logDensity(5);
        double expectedDensity = expectedDensity1 + expectedDensity12 + expectedDensity2 + expectedDensity22;

        double density = mvg.logPdf(DoubleTensor.create(new double[]{8, 10, 4, 5}, 2, 2));

        assertEquals(expectedDensity, density, 0.0001);
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
    public void dlogProbMatchesBivariateDiagonalGaussian() {
        UniformVertex mu = new UniformVertex(new long[]{2}, 0, 1);
        mu.setValue(DoubleTensor.create(-1, 2));
        UniformVertex sigma = new UniformVertex(new long[]{2}, 0, 1);
        sigma.setValue(DoubleTensor.create(0.5, 1.0));
        DoubleVertex covariance = sigma.pow(2).diag();

        GaussianVertex g = new GaussianVertex(mu, sigma);
        g.setValue(DoubleTensor.create(0.5, 3));
        MultivariateGaussianVertex mvg = new MultivariateGaussianVertex(mu, covariance);
        mvg.setValue(DoubleTensor.create(0.5, 3));

        LogProbGradientCalculator gradienOfG = new LogProbGradientCalculator(ImmutableList.of(g), ImmutableList.of(mu, sigma));
        LogProbGradientCalculator gradienOfMVG = new LogProbGradientCalculator(ImmutableList.of(mvg), ImmutableList.of(mu, sigma));
        Map<VertexId, DoubleTensor> gDLogProb = gradienOfG.getJointLogProbGradientWrtLatents();
        Map<VertexId, DoubleTensor> mvgDLogProb = gradienOfMVG.getJointLogProbGradientWrtLatents();

        assertThat(gDLogProb.get(mu.getId()), valuesAndShapesMatch(mvgDLogProb.get(mu.getId())));
        assertThat(gDLogProb.get(sigma.getId()), valuesAndShapesMatch(mvgDLogProb.get(sigma.getId())));
    }

    @Test
    public void dlogProbMatchesUnivariateGaussian() {
        UniformVertex mu = new UniformVertex(new long[]{1}, 0, 1);
        mu.setValue(DoubleTensor.create(-1));
        UniformVertex sigma = new UniformVertex(new long[]{1}, 0, 1);
        sigma.setValue(DoubleTensor.create(0.5));
        DoubleVertex variance = sigma.pow(2).diag();

        GaussianVertex g = new GaussianVertex(mu, sigma);
        g.setValue(DoubleTensor.create(0.5));
        MultivariateGaussianVertex mvg = new MultivariateGaussianVertex(mu, variance);
        mvg.setValue(DoubleTensor.create(0.5));

        LogProbGradientCalculator gradienOfG = new LogProbGradientCalculator(ImmutableList.of(g), ImmutableList.of(mu, sigma));
        LogProbGradientCalculator gradienOfMVG = new LogProbGradientCalculator(ImmutableList.of(mvg), ImmutableList.of(mu, sigma));
        Map<VertexId, DoubleTensor> gDLogProb = gradienOfG.getJointLogProbGradientWrtLatents();
        Map<VertexId, DoubleTensor> mvgDLogProb = gradienOfMVG.getJointLogProbGradientWrtLatents();

        assertThat(gDLogProb.get(mu.getId()), valuesAndShapesMatch(mvgDLogProb.get(mu.getId())));
        assertThat(gDLogProb.get(sigma.getId()), valuesAndShapesMatch(mvgDLogProb.get(sigma.getId())));
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPdCoVariance() {
        UniformVertex uniformA = new UniformVertex(new long[]{2}, -100, 100);
        MultivariateGaussianVertex mvg = new MultivariateGaussianVertex(ConstantVertex.of(-1., 2.), uniformA.diag());

        DoubleTensor vertexStartValue = DoubleTensor.create(-10, 3);
        DoubleTensor vertexEndValue = DoubleTensor.create(-9.5, 3.5);
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(
            DoubleTensor.create(0.1, 0.1),
            DoubleTensor.create(3.0, 3.0),
            0.1,
            uniformA,
            mvg,
            vertexStartValue,
            vertexEndValue,
            vertexIncrement,
            0.0001);
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPdMu() {
        UniformVertex uniformA = new UniformVertex(new long[]{2}, -100, 100);
        MultivariateGaussianVertex mvg = new MultivariateGaussianVertex(uniformA, ConstantVertex.of(0.5, 2).diag());

        DoubleTensor vertexStartValue = DoubleTensor.create(-10, 3);
        DoubleTensor vertexEndValue = DoubleTensor.create(-9.5, 3.5);
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(
            DoubleTensor.create(-3, -3),
            DoubleTensor.create(3.0, 3.0),
            0.1,
            uniformA,
            mvg,
            vertexStartValue,
            vertexEndValue,
            vertexIncrement,
            0.0001);
    }

    @Test
    public void gradientOfLogProbGraphMatchesUnivariateGaussian() {

        UniformVertex mu = new UniformVertex(new long[]{1}, 0, 1);
        mu.setValue(DoubleTensor.create(0));
        UniformVertex variance = new UniformVertex(new long[]{1, 1}, 0, 1);
        variance.setValue(DoubleTensor.create(1.5).reshape(1, 1));

        MultivariateGaussianVertex mvg = new MultivariateGaussianVertex(mu, variance);
        mvg.setValue(DoubleTensor.create(0.5));

        Map<Vertex, DoubleTensor> mvgDLogProb = mvg.dLogProbAtValue(mu, variance);

        LogProbGraph logProbGraph = mvg.logProbGraph();
        logProbGraph.getPlaceholder(mu).setValue(mu.getValue());
        logProbGraph.getPlaceholder(variance).setValue(variance.getValue());
        logProbGraph.getPlaceholder(mvg).setValue(mvg.getValue());

        PartialsOf partialsOf = Differentiator.reverseModeAutoDiff(
            logProbGraph.getLogProbOutput(),
            logProbGraph.getPlaceholder(mu),
            logProbGraph.getPlaceholder(variance)
        );

        DoubleTensor wrtMu = partialsOf.withRespectTo(logProbGraph.getPlaceholder(mu));
        DoubleTensor wrtSigma = partialsOf.withRespectTo(logProbGraph.getPlaceholder(variance));

        assertThat(wrtMu, valuesAndShapesMatch(mvgDLogProb.get(mu)));
        assertThat(wrtSigma, valuesAndShapesMatch(mvgDLogProb.get(variance)));
    }

    @Test
    public void infer1DCovarianceParamsFromSamples() {

        DoubleTensor trueCovarianceDiag = DoubleTensor.create(1.5);

        List<DoubleVertex> muCov = new ArrayList<>();
        muCov.add(ConstantVertex.of(trueCovarianceDiag));

        List<DoubleVertex> latentMuCov = new ArrayList<>();
        UniformVertex latentCovDiag = new UniformVertex(new long[]{2}, 0.01, 10.0);
        latentCovDiag.setAndCascade(DoubleTensor.create(0.5));
        latentMuCov.add(latentCovDiag);

        int numSamples = 200;
        VertexVariationalMAP.inferHyperParamsFromSamples(
            hyperParams -> new MultivariateGaussianVertex(new long[]{numSamples, 1}, ConstantVertex.of(DoubleTensor.create(-1)), hyperParams.get(0).diag()),
            muCov,
            latentMuCov,
            random
        );
    }


    @Test
    public void infer2DCovarianceParamsFromSamples() {

        DoubleTensor trueCovarianceDiag = DoubleTensor.create(1, 2);

        List<DoubleVertex> muCov = new ArrayList<>();
        muCov.add(ConstantVertex.of(trueCovarianceDiag));

        List<DoubleVertex> latentMuCov = new ArrayList<>();
        UniformVertex latentCovDiag = new UniformVertex(new long[]{2}, 0.01, 10.0);
        latentCovDiag.setAndCascade(DoubleTensor.create(4, 7));
        latentMuCov.add(latentCovDiag);

        int numSamples = 200;
        VertexVariationalMAP.inferHyperParamsFromSamples(
            hyperParams -> new MultivariateGaussianVertex(new long[]{numSamples, 2}, ConstantVertex.of(DoubleTensor.create(-1, 2)), hyperParams.get(0).diag()),
            muCov,
            latentMuCov,
            random
        );
    }

    @Test
    public void infer2DMuParamFromSamples() {

        DoubleTensor trueMu = DoubleTensor.create(-1, 2);
        List<DoubleVertex> muCov = new ArrayList<>();
        muCov.add(ConstantVertex.of(trueMu));

        List<DoubleVertex> latentMuCov = new ArrayList<>();
        UniformVertex latentMu = new UniformVertex(new long[]{2}, -10, 10.0);
        latentMu.setAndCascade(DoubleTensor.create(9, -9));
        latentMuCov.add(latentMu);

        int numSamples = 200;
        VertexVariationalMAP.inferHyperParamsFromSamples(
            hyperParams -> new MultivariateGaussianVertex(new long[]{numSamples, 2}, hyperParams.get(0), ConstantVertex.of(DoubleTensor.create(1, 2).diag())),
            muCov,
            latentMuCov,
            random
        );
    }

    @Test
    public void infer2DMuAndDiagonalCovarianceParamFromSamples() {

        DoubleTensor trueMu = DoubleTensor.create(-1, 2);
        DoubleTensor trueCovarianceDiag = DoubleTensor.create(1, 2);

        List<DoubleVertex> muCov = new ArrayList<>();
        muCov.add(ConstantVertex.of(trueMu));
        muCov.add(ConstantVertex.of(trueCovarianceDiag));

        List<DoubleVertex> latentMuCov = new ArrayList<>();
        UniformVertex latentMu = new UniformVertex(new long[]{2}, -100, 100.0);
        latentMu.setAndCascade(DoubleTensor.create(3, 3));
        latentMuCov.add(latentMu);

        UniformVertex latentCovDiag = new UniformVertex(new long[]{2}, 0.01, 100.0);
        latentCovDiag.setAndCascade(DoubleTensor.create(0.5, 1));
        latentMuCov.add(latentCovDiag);

        int numSamples = 200;
        VertexVariationalMAP.inferHyperParamsFromSamples(
            hyperParams -> new MultivariateGaussianVertex(new long[]{numSamples, 2}, hyperParams.get(0), hyperParams.get(1).diag()),
            muCov,
            latentMuCov,
            random
        );
    }

}