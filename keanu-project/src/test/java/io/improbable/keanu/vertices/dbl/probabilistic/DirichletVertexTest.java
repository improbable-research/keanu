package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import org.apache.commons.math3.util.Pair;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDoubleTensorContract.sampleMethodMatchesLogProbMethodMultiVariate;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.number.IsCloseTo.closeTo;
import static org.junit.Assert.assertEquals;

public class DirichletVertexTest {

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void twoDimensionalDirichletEqualsABeta() {
        double alpha = 0.4;
        double beta = 1.;
        BetaVertex betaVertex = new BetaVertex(alpha, beta);
        DirichletVertex dirichletVertex = new DirichletVertex(alpha, beta);

        Assert.assertEquals(betaVertex.logPdf(0.5), dirichletVertex.logPdf(DoubleTensor.create(new double[]{0.5, 0.5})), 1e-6);
        Assert.assertEquals(betaVertex.logPdf(0.75), dirichletVertex.logPdf(DoubleTensor.create(new double[]{0.75, 0.25})), 1e-6);
    }

    @Test
    public void flatUniformIfAllConcentrationValuesAreOne() {
        DirichletVertex dirichlet = new DirichletVertex(1, 1);

        double twoDimDirichletPdf1 = dirichlet.logPdf(DoubleTensor.create(new double[]{0.7, 0.3}));
        double twoDimDirichletPdf2 = dirichlet.logPdf(DoubleTensor.create(new double[]{0.3, 0.7}));
        double twoDimDirichletPdf3 = dirichlet.logPdf(DoubleTensor.create(new double[]{0.5, 0.5}));

        Assert.assertTrue(twoDimDirichletPdf1 == twoDimDirichletPdf2 && twoDimDirichletPdf2 == twoDimDirichletPdf3);

        dirichlet = new DirichletVertex(1, 1, 1, 1);

        double fourDimDirichletPdf1 = dirichlet.logPdf(DoubleTensor.create(new double[]{0.1, 0.2, 0.3, 0.4}));
        double fourDimDirichletPdf2 = dirichlet.logPdf(DoubleTensor.create(new double[]{0.7, 0.1, 0.1, 0.1}));
        double fourDimDirichletPdf3 = dirichlet.logPdf(DoubleTensor.create(new double[]{0.25, 0.25, 0.25, 0.25}));

        Assert.assertTrue(fourDimDirichletPdf1 == fourDimDirichletPdf2 && fourDimDirichletPdf2 == fourDimDirichletPdf3);
    }

    @Test
    public void matchesScipyDirichletLogPdf() {
        DirichletVertex dirichlet = new DirichletVertex(2, 2);
        double scipyPdf = 0.8662499999999997;
        Assert.assertEquals(scipyPdf, Math.exp(dirichlet.logPdf(new double[]{0.175, 0.825})), 1e-6);
    }

    @Test
    public void canSplitManyStringsOfVaryingSizeWithKnownMean() {
        DirichletVertex dirichlet = new DirichletVertex(10, 5, 3);
        int numSamples = 50000;
        DoubleTensor samples = Nd4jDoubleTensor.zeros(new int[]{numSamples, 3});

        for (int i = 0; i < numSamples; i++) {
            DoubleTensor sample = dirichlet.sample(random);
            samples.setValue(sample.getValue(0, 0), i, 0);
            samples.setValue(sample.getValue(0, 1), i, 1);
            samples.setValue(sample.getValue(0, 2), i, 2);
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
            bucketSize
        );
    }

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
        DoubleTensor startingValue = Nd4jDoubleTensor.create(new double[]{0.1, 0.9}, new int[]{1, 2});

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

            Map<Long, DoubleTensor> diffln = dirichlet.dLogProbAtValue();

            double actualDiff = diffln.get(concentrationHyperParam.getId()).getValue(0, 0) + diffln.get(concentrationHyperParam.getId()).getValue(0, 1);

            assertEquals(diffLnDensityApproxExpected, actualDiff, 0.001);

            startingValue = Nd4jDoubleTensor.create(new double[]{dirchletValue[0] + step, dirchletValue[1] - step}, new int[]{1, 2});
        }
    }

    private static void sampleMethodMatchesLogProbMethodMultiVariateDirichlet(Vertex<DoubleTensor> vertexUnderTest,
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
                Nd4jDoubleTensor bucket = new Nd4jDoubleTensor(new double[]{x1, x2, x3}, new int[]{1, 3});
                double densityAtBucketCenter = Math.exp(vertexUnderTest.logProb(bucket)) * bucketVolume;
                double actual = percentage;
                assertThat("Problem with logProb at " + bucketCenter, densityAtBucketCenter, closeTo(actual, maxError));
            }
        }

    }

}
