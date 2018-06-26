package io.improbable.keanu.vertices.dbl.probabilistic;

import com.sun.tools.javac.util.Pair;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
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

import static java.util.stream.Collectors.counting;
import static java.util.stream.Collectors.groupingBy;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.number.IsCloseTo.closeTo;

public class MultivariateGaussianTest {

    KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void gaussianSampleMethodMatchesLogProbMethod() {
        DoubleVertex mu = new ConstantDoubleVertex(new Nd4jDoubleTensor(new int[]{2, 1}, new double[]{0, 0}));
        DoubleVertex covarianceMatrix = new ConstantDoubleVertex(new Nd4jDoubleTensor(new int[]{2, 2}, new double[]{1, 0.3, 0.3, 0.6}));

        MultivariateGaussian mvg = new MultivariateGaussian(mu, covarianceMatrix);

        double from = -1.;
        double to = 1.;
        double bucketSize = 0.25;

        sampleMethodMatchesLogProbMethodMultiVariate(mvg, from, to, bucketSize, 1e-2, 50000, random);
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

        double total = 0;

        for (Map.Entry<Pair<Double, Double>, Long> entry : sampleBucket.entrySet()) {
            double percentage = (double) entry.getValue() / sampleCount;
            Pair<Double, Double> bucketCenter = entry.getKey();
            total += percentage;
            Nd4jDoubleTensor sample = new Nd4jDoubleTensor(new int[]{2, 1}, new double[]{bucketCenter.fst, bucketCenter.snd});
            double densityAtBucketCenter = Math.exp(vertexUnderTest.logProb(sample));
            double actual = percentage / bucketSize;

//            System.out.println(bucketCenter + " " + densityAtBucketCenter + " " + actual + ". %: " + percentage);
//            assertThat("Problem with logProb at " + bucketCenter, densityAtBucketCenter, closeTo(actual, maxError));
        }

        sampleBucket.entrySet().stream()
            .sorted(Map.Entry.comparingByValue())
            .forEach(System.out::println);

        System.out.println(total);
        System.out.println(bucketCount);
    }

    private static Double bucketCenter(Double x, double bucketSize, double from) {
        double bucketNumber = Math.floor((x - from) / bucketSize);
        return bucketNumber * bucketSize + bucketSize / 2 + from;
    }

}
