package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.testcategory.Slow;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.LogProbGraphContract;
import io.improbable.keanu.vertices.LogProbGraphValueFeeder;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.utility.GraphAssertionException;
import org.apache.commons.math3.distribution.BetaDistribution;
import org.apache.commons.math3.util.Pair;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.junit.rules.ExpectedException;
import umontreal.ssj.probdistmulti.DirichletDist;

import java.util.HashMap;
import java.util.Map;

import static io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDoubleTensorContract.sampleMethodMatchesLogProbMethodMultiVariate;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.greaterThan;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.number.IsCloseTo.closeTo;
import static org.junit.Assert.assertEquals;

public class DirichletVertexTest {

    private KeanuRandom random;

    @Rule
    public ExpectedException thrown = ExpectedException.none();

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void twoDimensionalDirichletLogProbEqualsABeta() {
        double alpha = 0.4;
        double beta = 1.;
        BetaVertex betaVertex = new BetaVertex(alpha, beta);
        DirichletVertex dirichletVertex = new DirichletVertex(alpha, beta);

        Assert.assertEquals(betaVertex.logPdf(0.5), dirichletVertex.logPdf(DoubleTensor.create(0.5, 0.5)), 1e-6);
        Assert.assertEquals(betaVertex.logPdf(0.75), dirichletVertex.logPdf(DoubleTensor.create(0.75, 0.25)), 1e-6);
        Assert.assertEquals(betaVertex.logPdf(0.0), dirichletVertex.logPdf(DoubleTensor.create(0.0, 1)), 1e-6);
    }

    @Test
    public void twoDimensionalDirichletLogProbGraphEqualsABeta() {
        DoubleVertex concentration = ConstantVertex.of(0.4, 1.);
        DirichletVertex vertex = new DirichletVertex(concentration);
        BetaDistribution betaDistribution = new BetaDistribution(0.4, 1.);
        LogProbGraph logProbGraph = vertex.logProbGraph();
        LogProbGraphValueFeeder.feedValue(logProbGraph, concentration, concentration.getValue());

        LogProbGraphValueFeeder.feedValue(logProbGraph, vertex, DoubleTensor.create(0.5, 0.5));
        double expectedDensity = betaDistribution.logDensity(0.5);
        LogProbGraphContract.matchesKnownLogDensity(logProbGraph, expectedDensity);

        LogProbGraphValueFeeder.feedValueAndCascade(logProbGraph, vertex, DoubleTensor.create(0.75, 0.25));
        expectedDensity = betaDistribution.logDensity(0.75);
        LogProbGraphContract.matchesKnownLogDensity(logProbGraph, expectedDensity);
    }

    @Test
    public void logProbIsFlatUniformIfAllConcentrationValuesAreOne() {
        DirichletVertex dirichlet = new DirichletVertex(1, 1);

        double twoDimDirichletPdf1 = dirichlet.logPdf(DoubleTensor.create(0.7, 0.3));
        double twoDimDirichletPdf2 = dirichlet.logPdf(DoubleTensor.create(0.3, 0.7));
        double twoDimDirichletPdf3 = dirichlet.logPdf(DoubleTensor.create(0.5, 0.5));

        Assert.assertTrue(twoDimDirichletPdf1 == twoDimDirichletPdf2 && twoDimDirichletPdf2 == twoDimDirichletPdf3);

        dirichlet = new DirichletVertex(new long[]{1, 4}, 1);

        double fourDimDirichletPdf1 = dirichlet.logPdf(DoubleTensor.create(0.1, 0.2, 0.3, 0.4));
        double fourDimDirichletPdf2 = dirichlet.logPdf(DoubleTensor.create(0.7, 0.1, 0.1, 0.1));
        double fourDimDirichletPdf3 = dirichlet.logPdf(DoubleTensor.create(0.25, 0.25, 0.25, 0.25));

        Assert.assertTrue(fourDimDirichletPdf1 == fourDimDirichletPdf2 && fourDimDirichletPdf2 == fourDimDirichletPdf3);
    }

    @Test
    public void logProbGraphIsFlatUniformIfAllConcentrationValuesAreOne() {
        DoubleVertex concentration = ConstantVertex.of(DoubleTensor.create(1, new long[]{1, 4}));
        DirichletVertex dirichlet = new DirichletVertex(concentration);

        LogProbGraph logProbGraph1 = dirichlet.logProbGraph();
        LogProbGraphValueFeeder.feedValue(logProbGraph1, concentration, concentration.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph1, dirichlet, DoubleTensor.create(0.1, 0.2, 0.3, 0.4));

        LogProbGraph logProbGraph2 = dirichlet.logProbGraph();
        LogProbGraphValueFeeder.feedValue(logProbGraph2, concentration, concentration.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph2, dirichlet, DoubleTensor.create(0.7, 0.1, 0.1, 0.1));

        LogProbGraph logProbGraph3 = dirichlet.logProbGraph();
        LogProbGraphValueFeeder.feedValue(logProbGraph3, concentration, concentration.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph3, dirichlet, DoubleTensor.create(0.25, 0.25, 0.25, 0.25));

        LogProbGraphContract.equal(logProbGraph1, logProbGraph2);
        LogProbGraphContract.equal(logProbGraph2, logProbGraph3);
    }

    @Test
    public void logProbMatchesMontrealDirichletLogPdf() {
        DirichletDist baseline = new DirichletDist(new double[]{3, 4, 5});
        DirichletVertex keanu = new DirichletVertex(3, 4, 5);

        Assert.assertEquals(Math.log(baseline.density(new double[]{0.1, 0.6, 0.3})),
            keanu.logPdf(new double[]{0.1, 0.6, 0.3}), 1e-6);
        Assert.assertEquals(Math.log(baseline.density(new double[]{0.3, 0.4, 0.3})),
            keanu.logPdf(new double[]{0.3, 0.4, 0.3}), 1e-6);
    }

    @Test
    public void logProbGraphMatchesMontrealDirichletLogPdf() {
        DoubleVertex concentration = ConstantVertex.of(3., 4., 5.);
        DirichletVertex vertex = new DirichletVertex(concentration);
        DirichletDist baseline = new DirichletDist(new double[]{3, 4, 5});
        LogProbGraph logProbGraph = vertex.logProbGraph();
        LogProbGraphValueFeeder.feedValue(logProbGraph, concentration, concentration.getValue());

        LogProbGraphValueFeeder.feedValue(logProbGraph, vertex, DoubleTensor.create(0.1, 0.6, 0.3));
        double expectedDensity = Math.log(baseline.density(new double[]{0.1, 0.6, 0.3}));
        LogProbGraphContract.matchesKnownLogDensity(logProbGraph, expectedDensity);

        LogProbGraphValueFeeder.feedValueAndCascade(logProbGraph, vertex, DoubleTensor.create(0.3, 0.4, 0.3));
        expectedDensity = Math.log(baseline.density(new double[]{0.3, 0.4, 0.3}));
        LogProbGraphContract.matchesKnownLogDensity(logProbGraph, expectedDensity);
    }


    @Test
    public void logProbGraphThrowsExceptionIfSumOfXIsNotEqualTo1WithEpsilon() {
        DoubleVertex concentration = ConstantVertex.of(3., 4., 5.);
        DirichletVertex vertex = new DirichletVertex(concentration);
        DoubleTensor x = DoubleTensor.create(0.1, 0.6, 0.300011);
        assertThat(Math.abs(x.sum() - 1.), is(greaterThan((.00001))));

        LogProbGraph logProbGraph = vertex.logProbGraph();
        LogProbGraphValueFeeder.feedValue(logProbGraph, concentration, concentration.getValue());

        thrown.expect(GraphAssertionException.class);
        thrown.expectMessage("Sum of values to calculate Dirichlet likelihood for must equal 1");

        LogProbGraphValueFeeder.feedValueAndCascade(logProbGraph, vertex, x);
    }

    @Category(Slow.class)
    @Test
    public void canSplitManyStringsOfVaryingSizeWithKnownMean() {
        DirichletVertex dirichlet = new DirichletVertex(10, 5, 3);
        int numSamples = 30000;
        DoubleTensor samples = DoubleTensor.zeros(new long[]{numSamples, 3});

        for (int i = 0; i < numSamples; i++) {
            DoubleTensor sample = dirichlet.sample(random);
            samples.setValue(sample.getValue(0), i, 0);
            samples.setValue(sample.getValue(1), i, 1);
            samples.setValue(sample.getValue(2), i, 2);
        }

        DoubleTensor stringOne = samples.slice(1, 0);
        DoubleTensor stringTwo = samples.slice(1, 1);
        DoubleTensor stringThree = samples.slice(1, 2);

        double stringOneLength = stringOne.average();
        double stringTwoLength = stringTwo.average();
        double stringThreeLength = stringThree.average();

        Assert.assertEquals(1.0, stringOneLength + stringTwoLength + stringThreeLength, 1e-3);
        Assert.assertEquals(10. / 18., stringOneLength, 1e-3);
        Assert.assertEquals(5. / 18., stringTwoLength, 1e-3);
        Assert.assertEquals(3. / 18., stringThreeLength, 1e-3);
    }

    @Category(Slow.class)
    @Test
    public void twoDimensionalDirichletSampleMethodMatchesLogProbMethod() {
        DirichletVertex dirichlet = new DirichletVertex(5, 5);

        double from = 0.1;
        double to = 0.9;
        double bucketSize = 0.05;

        sampleMethodMatchesLogProbMethodMultiVariate(
            dirichlet,
            from,
            to,
            bucketSize,
            0.01,
            10000,
            random,
            bucketSize,
            true
        );
    }

    @Category(Slow.class)
    @Test
    public void threeDimensionalDirichletSampleMethodMatchesLogProbMethod() {
        DirichletVertex dirichlet = new DirichletVertex(2, 2, 2);

        double from = 0.1;
        double to = 0.9;
        double bucketSize = 0.05;

        sampleMethodMatchesLogProbMethodMultiVariateDirichlet(
            dirichlet,
            from,
            to,
            bucketSize,
            0.01,
            10000,
            random,
            bucketSize * bucketSize
        );
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPdconcentration() {
        UniformVertex concentrationHyperParam = new UniformVertex(1.5, 3.0);
        DoubleTensor hyperParamValue = DoubleTensor.create(new double[]{7, 7}, 1, 2);
        concentrationHyperParam.setValue(hyperParamValue);

        DirichletVertex dirichlet = new DirichletVertex(concentrationHyperParam);
        DoubleTensor startingValue = DoubleTensor.create(new double[]{0.1, 0.9}, new long[]{1, 2});

        double start = 0.1;
        double end = 0.9;
        double step = 0.05;
        double gradientDelta = 0.01;

        for (double i = start; i < end; i = i + step) {

            dirichlet.setAndCascade(startingValue);
            double[] dirchletValue = startingValue.asFlatDoubleArray();

            double[] values = hyperParamValue.asFlatDoubleArray();
            concentrationHyperParam.setAndCascade(DoubleTensor.create(new double[]{values[0] - gradientDelta, values[1] - gradientDelta}));
            double lnDensityA1 = dirichlet.logProb(startingValue);

            concentrationHyperParam.setAndCascade(DoubleTensor.create(new double[]{values[0] + gradientDelta, values[1] + gradientDelta}));
            double lnDensityA2 = dirichlet.logProb(startingValue);

            double diffLnDensityApproxExpected = (lnDensityA2 - lnDensityA1) / (2 * gradientDelta);

            Map<Vertex, DoubleTensor> diffln = dirichlet.dLogProbAtValue(concentrationHyperParam);

            double actualDiff = diffln.get(concentrationHyperParam).getValue(0, 0) + diffln.get(concentrationHyperParam).getValue(0, 1);

            assertEquals(diffLnDensityApproxExpected, actualDiff, 0.001);

            startingValue = DoubleTensor.create(new double[]{dirchletValue[0] + step, dirchletValue[1] - step}, new long[]{1, 2});
        }
    }

    private static <V extends Vertex<DoubleTensor> & Probabilistic<DoubleTensor>> void sampleMethodMatchesLogProbMethodMultiVariateDirichlet(V vertexUnderTest,
                                                                                                                                             double from,
                                                                                                                                             double to,
                                                                                                                                             double bucketSize,
                                                                                                                                             double maxError,
                                                                                                                                             int sampleCount,
                                                                                                                                             KeanuRandom random,
                                                                                                                                             double bucketVolume) {
        double bucketCount = ((to - from) / bucketSize);
        double halfBucket = bucketSize / 2;

        if (bucketCount != (int) bucketCount) {
            throw new IllegalArgumentException("Range must be evenly divisible by bucketSize");
        }

        double[][] samples = new double[sampleCount][2];

        for (int i = 0; i < sampleCount; i++) {
            DoubleTensor sample = vertexUnderTest.sample(random);
            samples[i] = sample.asFlatDoubleArray();
        }

        Map<Pair<Double, Double>, Long> sampleBucket = new HashMap<>();

        for (double firstDimension = from; firstDimension < to; firstDimension = firstDimension + bucketSize) {
            for (double secondDimension = from; secondDimension < to; secondDimension = secondDimension + bucketSize) {
                sampleBucket.put(new Pair<>(firstDimension + halfBucket, secondDimension + halfBucket), 0L);
            }
        }

        for (int i = 0; i < sampleCount; i++) {
            double sampleX = samples[i][0];
            double sampleY = samples[i][1];
            for (Pair<Double, Double> bucketCenter : sampleBucket.keySet()) {

                if (sampleX > bucketCenter.getFirst() - halfBucket
                    && sampleX < bucketCenter.getFirst() + halfBucket
                    && sampleY > bucketCenter.getSecond() - halfBucket
                    && sampleY < bucketCenter.getSecond() + halfBucket) {
                    sampleBucket.put(bucketCenter, sampleBucket.get(bucketCenter) + 1);
                    break;
                }

            }
        }

        for (Map.Entry<Pair<Double, Double>, Long> entry : sampleBucket.entrySet()) {
            double percentage = (double) entry.getValue() / sampleCount;
            if (percentage != 0) {
                double[] bucketCenter = new double[]{entry.getKey().getFirst(), entry.getKey().getSecond()};
                double x1 = bucketCenter[0];
                double x2 = bucketCenter[1];
                if (x1 + x2 > 1.) {
                    break;
                }
                double x3 = 1 - x1 - x2;
                DoubleTensor bucket = DoubleTensor.create(new double[]{x1, x2, x3}, new long[]{1, 3});
                double densityAtBucketCenter = Math.exp(vertexUnderTest.logProb(bucket)) * bucketVolume;
                double actual = percentage;
                assertThat("Problem with logProb at " + bucketCenter, densityAtBucketCenter, closeTo(actual, maxError));
            }
        }

    }

}
