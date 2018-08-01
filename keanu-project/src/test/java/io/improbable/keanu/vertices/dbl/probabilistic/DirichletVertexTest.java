package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import org.apache.commons.math3.util.Pair;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.number.IsCloseTo.closeTo;

public class DirichletVertexTest {

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void sampleFromUnivariateReturnsFlatDistribution() {
        DirichletVertex dirichlet = new DirichletVertex(2.0);

        Assert.assertEquals(1.0, dirichlet.sample(random).scalar(), 1e-6);
        Assert.assertEquals(1.0, dirichlet.logPdf(0.1), 1e-6);
        Assert.assertEquals(1.0, dirichlet.logPdf(0.5), 1e-6);
        Assert.assertEquals(1.0, dirichlet.logPdf(0.9), 1e-6);
    }

    @Test
    public void flatDirichletIfAllConcentrationAreOnes() {
        DirichletVertex dirichlet = new DirichletVertex(new ConstantDoubleVertex(new double[]{1, 1}));

        Assert.assertEquals(0.0, dirichlet.logPdf(DoubleTensor.create(new double[]{1.3, 1.6})), 1e-6);
        Assert.assertEquals(0.0, dirichlet.logPdf(DoubleTensor.create(new double[]{0.3, 0.6})), 1e-6);
        Assert.assertEquals(0.0, dirichlet.logPdf(DoubleTensor.create(new double[]{30, 50})), 1e-6);
    }

    @Test
    public void matchesScip() {
        DirichletVertex dirichlet = new DirichletVertex(new ConstantDoubleVertex(new double[]{2, 2}));
        double scipyAnswer = 0.8662499999999997;
        Assert.assertEquals(scipyAnswer, Math.exp(dirichlet.logPdf(new double[]{0.175, 0.825})), 1e-6);
        Assert.assertEquals(scipyAnswer, Math.exp(dirichlet.logPdf(new double[]{0.175, 0.825})), 1e-6);
        Assert.assertEquals(scipyAnswer, Math.exp(dirichlet.logPdf(new double[]{0.175, 0.825})), 1e-6);
    }

    @Test
    public void splitStrings() {
        DirichletVertex dirichlet = new DirichletVertex(new ConstantDoubleVertex(new double[]{10, 5, 3}));
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
    public void dirichletSampleMethodMatchesLogProbMethod() {
        DirichletVertex mvg = new DirichletVertex(new ConstantDoubleVertex(new double[]{5, 5}));

        double from = 0.1;
        double to = 0.9;
        double bucketSize = 0.05;

        sampleMethodMatchesLogProbMethodMultiVariate(mvg, from, to, bucketSize, 0.01, 100000, random);
    }

    private static void sampleMethodMatchesLogProbMethodMultiVariate(DirichletVertex vertexUnderTest,
                                                                     double from,
                                                                     double to,
                                                                     double bucketSize,
                                                                     double maxError,
                                                                     int sampleCount,
                                                                     KeanuRandom random) {
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
                Nd4jDoubleTensor bucket = new Nd4jDoubleTensor(bucketCenter, new int[]{1, 2});
                double densityAtBucketCenter = Math.exp(vertexUnderTest.logProb(bucket)) * bucketSize;
                double actual = (percentage);
                assertThat("Problem with logProb at " + bucketCenter, densityAtBucketCenter, closeTo(actual, maxError));
            }
        }
    }
}
