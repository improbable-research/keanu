package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.distributions.continuous.Gaussian;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import org.apache.commons.math3.util.Pair;
import org.junit.Before;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDoubleTensorContract.sampleUnivariateMethodMatchesLogProbMethod;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.number.IsCloseTo.closeTo;
import static org.junit.Assert.assertEquals;

public class MultivariateGaussianTest {

    KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void samplingFromUnivariateGaussianMatchesLogDensity() {
        MultivariateGaussian mvg = new MultivariateGaussian(0, 1);

        double from = -2.;
        double to = 2.;
        double bucketSize = 0.05;

        sampleUnivariateMethodMatchesLogProbMethod(mvg, from, to, bucketSize, 1e-2, random, 1000000);
    }

    @Test
    public void univariateGaussianMatchesLogDensityOfScalar() {
        MultivariateGaussian mvg = new MultivariateGaussian(5, 1);

        double expectedDensity = Gaussian.logPdf(5.0, 1, 0.5);
        double density = mvg.logPdf(Nd4jDoubleTensor.scalar(0.5));

        assertEquals(expectedDensity, density, 1e-2);
    }

    @Test
    public void bivariateGaussianMatchesLogDensityOfVector() {
        DoubleVertex mu = ConstantVertex.of(
            new Nd4jDoubleTensor(new int[]{2, 1}, new double[]{2, 3}));

        MultivariateGaussian mvg = new MultivariateGaussian(mu, 1);

        double expectedDensity1 = Gaussian.logPdf(2.0, 1, 8);
        double expectedDensity2 = Gaussian.logPdf(3.0, 1, 10);
        double expectedDensity = expectedDensity1 + expectedDensity2;

        double density = mvg.logPdf(new Nd4jDoubleTensor(new int[]{2, 1}, new double[]{8, 10}));

        assertEquals(expectedDensity, density, 0.0001);
    }

    @Test
    public void bivariateGaussianMatchesLogDensityOfScipy() {
        DoubleVertex mu = ConstantVertex.of(
            new Nd4jDoubleTensor(new int[]{2, 1}, new double[]{1, 2}));
        DoubleVertex covarianceMatrix = ConstantVertex.of(
            new Nd4jDoubleTensor(new int[]{2, 2}, new double[]{1, 0.3, 0.3, 0.6}));

        MultivariateGaussian mvg = new MultivariateGaussian(mu, covarianceMatrix);
        double density = mvg.logPdf(new Nd4jDoubleTensor(new int[]{2, 1}, new double[]{0.5, 0.4}));
        double expected = -3.6874792995813834;

        assertEquals(expected, density, 0.001);
    }

    @Test
    public void multivariateGaussianMatchesLogDensityOfScipy() {
        DoubleVertex mu = ConstantVertex.of(
            new Nd4jDoubleTensor(new int[]{3, 1}, new double[]{1, 2, 3}));

        DoubleVertex covarianceMatrix = ConstantVertex.of(
            new Nd4jDoubleTensor(
                new int[]{3, 3}, new double[]{
                1.0, 0.3, 0.3,
                0.3, 0.8, 0.3,
                0.3, 0.3, 0.6
            }
            )
        );

        MultivariateGaussian mvg = new MultivariateGaussian(mu, covarianceMatrix);
        double density = mvg.logPdf(new Nd4jDoubleTensor(new int[]{3, 1}, new double[]{0.2, 0.3, 0.4}));
        double expected = -8.155504532016181;

        assertEquals(expected, density, 0.001);
    }

    @Test
    public void gaussianSampleMethodMatchesLogProbMethod() {
        DoubleVertex mu = ConstantVertex.of(
            new Nd4jDoubleTensor(new int[]{2, 1}, new double[]{0, 0}));

        MultivariateGaussian mvg = new MultivariateGaussian(mu, 1);

        double from = -1.;
        double to = 1.;
        double bucketSize = 0.05;

        sampleMethodMatchesLogProbMethodMultiVariate(mvg, from, to, bucketSize, 0.01, 100000, random);
    }

    private static void sampleMethodMatchesLogProbMethodMultiVariate(MultivariateGaussian vertexUnderTest,
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
            double[] bucketCenter = new double[]{entry.getKey().getFirst(), entry.getKey().getSecond()};
            Nd4jDoubleTensor bucket = new Nd4jDoubleTensor(new int[]{2, 1}, bucketCenter);
            double densityAtBucketCenter = Math.exp(vertexUnderTest.logProb(bucket)) * bucketSize;
            double actual = (percentage / bucketSize);
            assertThat("Problem with logProb at " + bucketCenter, densityAtBucketCenter, closeTo(actual, maxError));
        }

    }
}
