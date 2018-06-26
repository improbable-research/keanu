package io.improbable.keanu.vertices.dbl.probabilistic;

import com.sun.tools.javac.util.Pair;
import io.improbable.keanu.distributions.continuous.Gaussian;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.tensor.dbl.ScalarDoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Stream;

import static io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDoubleTensorContract.sampleUnivariateMethodMatchesLogProbMethod;
import static java.util.stream.Collectors.counting;
import static java.util.stream.Collectors.groupingBy;
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
    public void samplingFromUnivariateNormalMatchesLogPdf() {
        DoubleVertex mu = new ConstantDoubleVertex(new Nd4jDoubleTensor(new int[]{1, 1}, new double[]{0}));
        DoubleVertex covarianceMatrix = new ConstantDoubleVertex(new Nd4jDoubleTensor(new int[]{1, 1}, new double[]{1}));

        MultivariateGaussian mvg = new MultivariateGaussian(mu, covarianceMatrix);

        double from = -2.;
        double to = 2.;
        double bucketSize = 0.05;

        sampleUnivariateMethodMatchesLogProbMethod(mvg, from, to, bucketSize, 1e-2, random, 1000000);
    }

    @Test
    public void univariateGaussianMatchesLogDensityOfScalar() {
        DoubleVertex mu = new ConstantDoubleVertex(new ScalarDoubleTensor(5.0));
        DoubleVertex covarianceMatrix = new ConstantDoubleVertex(new ScalarDoubleTensor(1));

        MultivariateGaussian mvg = new MultivariateGaussian(mu, covarianceMatrix);

        double expectedDensity = Gaussian.logPdf(5.0, 1, 0.5);
        double density = mvg.logPdf(Nd4jDoubleTensor.scalar(0.5));

        assertEquals(expectedDensity, density, 0.0001);
    }

    @Test
    public void bivariateGaussianMatchesLogDensityOfVector() {
        DoubleVertex mu = new ConstantDoubleVertex(new Nd4jDoubleTensor(new int[]{2, 1}, new double[]{2, 3}));
        DoubleVertex covarianceMatrix = new ConstantDoubleVertex(new Nd4jDoubleTensor(new int[]{2, 2}, new double[]{1, 0, 0, 1}));

        MultivariateGaussian mvg = new MultivariateGaussian(mu, covarianceMatrix);

        double expectedDensity1 = Gaussian.logPdf(2.0, 1, 8);
        double expectedDensity2 = Gaussian.logPdf(3.0, 1, 10);

        double density = mvg.logPdf(new Nd4jDoubleTensor(new int[]{2, 1}, new double[]{8, 10}));

        assertEquals(expectedDensity1 + expectedDensity2, density, 0.0001);
    }

    @Test
    public void bivariateGaussianMatchesLogDensityOfScipy() {
        DoubleVertex mu = new ConstantDoubleVertex(new Nd4jDoubleTensor(new int[]{2, 1}, new double[]{1, 2}));
        DoubleVertex covarianceMatrix = new ConstantDoubleVertex(new Nd4jDoubleTensor(new int[]{2, 2}, new double[]{1, 0.3, 0.3, 0.6}));

        MultivariateGaussian mvg = new MultivariateGaussian(mu, covarianceMatrix);

        double density = mvg.logPdf(new Nd4jDoubleTensor(new int[]{2, 1}, new double[]{0.5, 0.4}));

        double expected = -3.6874792995813834;

        assertEquals(expected, density, 0.001);
    }

    @Test
    public void multivariateGaussianMatchesLogDensityOfScipy() {
        DoubleVertex mu = new ConstantDoubleVertex(new Nd4jDoubleTensor(new int[]{3, 1}, new double[]{1, 2, 3}));

        DoubleVertex covarianceMatrix = new ConstantDoubleVertex(
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
        DoubleVertex mu = new ConstantDoubleVertex(new Nd4jDoubleTensor(new int[]{2, 1}, new double[]{0, 0}));
        DoubleVertex covarianceMatrix = new ConstantDoubleVertex(new Nd4jDoubleTensor(new int[]{2, 2}, new double[]{1, 0.0, 0.0, 1}));

        MultivariateGaussian mvg = new MultivariateGaussian(mu, covarianceMatrix);

        double from = -1.;
        double to = 1.;
        double bucketSize = 0.05;

        sampleMethodMatchesLogProbMethodMultiVariate(mvg, from, to, bucketSize, 0.1, 75000, random);
    }

    private static void sampleMethodMatchesLogProbMethodMultiVariate(MultivariateGaussian vertexUnderTest,
                                                        double from,
                                                        double to,
                                                        double bucketSize,
                                                        double maxError,
                                                        int sampleCount,
                                                        KeanuRandom random) {
        double bucketCount = ((to - from) / bucketSize);

        if (bucketCount != (int) bucketCount) {
            throw new IllegalArgumentException("Range must be evenly divisible by bucketSize");
        }

        double[][] samples = new double[sampleCount][2];

        for (int i = 0; i < sampleCount; i++) {
            DoubleTensor sample = vertexUnderTest.sample(random);
            samples[i] = sample.asFlatDoubleArray();
        }

        Map<Pair<Double, Double>, Long> sampleBucket = new HashMap<>();

        for (double foo = from; foo < to; foo=foo+bucketSize) {
            for (double bar = from; bar < to; bar=bar+bucketSize) {
                sampleBucket.put(new Pair<>(foo + bucketSize / 2, bar + bucketSize / 2), 0L);
            }
        }

        for (int i = 0; i < sampleCount; i++) {
            double sample1D = samples[i][0];
            double sample2D = samples[i][1];
            double halfBucket = 0.5 * bucketSize;
            for (Pair<Double, Double> bucketCenter : sampleBucket.keySet()) {
                if (sample1D > bucketCenter.fst - halfBucket && sample1D < bucketCenter.fst + halfBucket && sample2D > bucketCenter.snd - halfBucket && sample2D < bucketCenter.snd + halfBucket) {
                    sampleBucket.put(bucketCenter, sampleBucket.get(bucketCenter) + 1);
                    break;
                }
            }
        }

        for (Map.Entry<Pair<Double, Double>, Long> entry : sampleBucket.entrySet()) {
            double percentage = (double) entry.getValue() / sampleCount;
            Pair<Double, Double> bucketCenter = entry.getKey();
            Nd4jDoubleTensor sample = new Nd4jDoubleTensor(new int[]{2, 1}, new double[]{bucketCenter.fst, bucketCenter.snd});
            double densityAtBucketCenter = Math.exp(vertexUnderTest.logProb(sample));
            double actual = percentage / bucketSize;

            assertThat("Problem with logProb at " + bucketCenter, densityAtBucketCenter, closeTo(actual, maxError));
        }

    }

    private static Double bucketCenter(Double x, double bucketSize, double from) {
        double bucketNumber = Math.floor((x - from) / bucketSize);
        return bucketNumber * bucketSize + bucketSize / 2 + from;
    }

}
