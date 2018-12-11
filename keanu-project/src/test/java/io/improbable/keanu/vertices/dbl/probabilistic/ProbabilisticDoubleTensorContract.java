package io.improbable.keanu.vertices.dbl.probabilistic;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.Differentiator;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.commons.math3.util.Pair;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

import static java.util.stream.Collectors.counting;
import static java.util.stream.Collectors.groupingBy;
import static java.util.stream.Collectors.toMap;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.number.IsCloseTo.closeTo;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

public class ProbabilisticDoubleTensorContract {

    /**
     * This method brute force verifies that a given vertex's sample method accurately reflects its logProb method.
     * This is done for a given range with a specified resolution (bucketSize). The error due to the approximate
     * nature of the brute force technique will be larger where the gradient of the logProb is large as well. This
     * only works with a scalar value vertex due to logProb being an aggregation of all values.
     *
     * @param vertexUnderTest
     * @param from
     * @param to
     * @param bucketSize
     */
    public static <V extends DoubleVertex & ProbabilisticDouble>
    void sampleMethodMatchesLogProbMethod(V vertexUnderTest,
                                          double from,
                                          double to,
                                          double bucketSize,
                                          double maxError,
                                          KeanuRandom random) {
        double bucketCount = ((to - from) / bucketSize);

        if (bucketCount != (int) bucketCount) {
            throw new IllegalArgumentException("Range must be evenly divisible by bucketSize");
        }

        double[] samples = vertexUnderTest.sample(random).asFlatDoubleArray();

        Map<Double, Long> histogram = Arrays.stream(vertexUnderTest.sample(random).asFlatDoubleArray())
            .filter(value -> value >= from && value <= to)
            .boxed()
            .collect(groupingBy(
                x -> bucketCenter(x, bucketSize, from),
                counting()
            ));

        for (Map.Entry<Double, Long> sampleBucket : histogram.entrySet()) {
            double percentage = (double) sampleBucket.getValue() / samples.length;
            double bucketCenter = sampleBucket.getKey();

            double densityAtBucketCenter = Math.exp(vertexUnderTest.logProb(Nd4jDoubleTensor.scalar(bucketCenter)));
            double actual = percentage / bucketSize;
            assertThat("Problem with logProb at " + bucketCenter, densityAtBucketCenter, closeTo(actual, maxError));
        }
    }

    public static <V extends DoubleVertex & ProbabilisticDouble>
    void sampleUnivariateMethodMatchesLogProbMethod(V vertexUnderTest,
                                                    double from,
                                                    double to,
                                                    double bucketSize,
                                                    double maxError,
                                                    KeanuRandom random,
                                                    int sampleCount) {
        double bucketCount = ((to - from) / bucketSize);

        if (bucketCount != (int) bucketCount) {
            throw new IllegalArgumentException("Range must be evenly divisible by bucketSize");
        }

        double[] samples = new double[sampleCount];
        for (int i = 0; i < sampleCount; i++) {
            samples[i] = vertexUnderTest.sample(random).scalar();
        }

        Map<Double, Long> histogram = Arrays.stream(samples)
            .filter(value -> value >= from && value <= to)
            .boxed()
            .collect(groupingBy(
                x -> bucketCenter(x, bucketSize, from),
                counting()
            ));

        for (Map.Entry<Double, Long> sampleBucket : histogram.entrySet()) {
            double percentage = (double) sampleBucket.getValue() / samples.length;
            double bucketCenter = sampleBucket.getKey();

            double densityAtBucketCenter = Math.exp(vertexUnderTest.logProb(Nd4jDoubleTensor.scalar(bucketCenter)));
            double actual = percentage / bucketSize;
            assertThat("Problem with logProb at " + bucketCenter, densityAtBucketCenter, closeTo(actual, maxError));
        }
    }

    private static Double bucketCenter(Double x, double bucketSize, double from) {
        double bucketNumber = Math.floor((x - from) / bucketSize);
        return bucketNumber * bucketSize + bucketSize / 2 + from;
    }

    public static <V extends DoubleVertex & ProbabilisticDouble>
    void samplingProducesRealisticMeanAndStandardDeviation(int numberOfSamples,
                                                           V vertexUnderTest,
                                                           double expectedMean,
                                                           double expectedStandardDeviation,
                                                           double maxError,
                                                           KeanuRandom random) {
        List<Double> samples = new ArrayList<>();

        for (int i = 0; i < numberOfSamples; i++) {
            double sample = vertexUnderTest.sample(random).scalar();
            samples.add(sample);
        }

        SummaryStatistics stats = new SummaryStatistics();
        samples.forEach(stats::addValue);

        double mean = stats.getMean();
        double sd = stats.getStandardDeviation();

        assertThat("Problem with mean", expectedMean, closeTo(mean, maxError));
        assertThat("Problem with standard deviation", expectedStandardDeviation, closeTo(sd, maxError));
    }

    public static <V extends DoubleVertex & ProbabilisticDouble>
    void moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(DoubleTensor hyperParameterStartValue,
                                                                            DoubleTensor hyperParameterEndValue,
                                                                            double hyperParameterValueIncrement,
                                                                            DoubleVertex hyperParameterVertex,
                                                                            V vertexUnderTest,
                                                                            DoubleTensor vertexStartValue,
                                                                            DoubleTensor vertexEndValue,
                                                                            double vertexValueIncrement,
                                                                            double gradientDelta) {

        for (DoubleTensor value = vertexStartValue; value.scalar() <= vertexEndValue.scalar(); value.plusInPlace(vertexValueIncrement)) {
            vertexUnderTest.setAndCascade(value);
            testGradientAcrossMultipleHyperParameterValues(
                hyperParameterStartValue,
                hyperParameterEndValue,
                hyperParameterValueIncrement,
                hyperParameterVertex,
                vertexUnderTest,
                gradientDelta
            );
        }
    }

    public static void testGradientAcrossMultipleHyperParameterValues(DoubleTensor hyperParameterStartValue,
                                                                      DoubleTensor hyperParameterEndValue,
                                                                      double hyperParameterValueIncrement,
                                                                      DoubleVertex hyperParameterVertex,
                                                                      Probabilistic<?> vertexUnderTest,
                                                                      double gradientDelta) {

        for (DoubleTensor parameterValue = hyperParameterStartValue; parameterValue.scalar() <= hyperParameterEndValue.scalar(); parameterValue.plusInPlace(hyperParameterValueIncrement)) {
            testGradientAtHyperParameterValue(
                parameterValue,
                hyperParameterVertex,
                vertexUnderTest,
                gradientDelta
            );
        }
    }

    public static void testGradientAtHyperParameterValue(DoubleTensor hyperParameterValue,
                                                         DoubleVertex hyperParameterVertex,
                                                         Probabilistic<?> vertexUnderTest,
                                                         double gradientDelta) {

        hyperParameterVertex.setAndCascade(hyperParameterValue.minus(gradientDelta));
        double lnDensityA1 = vertexUnderTest.logProbAtValue();

        hyperParameterVertex.setAndCascade(hyperParameterValue.plus(gradientDelta));
        double lnDensityA2 = vertexUnderTest.logProbAtValue();

        double diffLnDensityApproxExpected = 0.0;
        if (lnDensityA1 != lnDensityA2) {
            diffLnDensityApproxExpected = (lnDensityA2 - lnDensityA1) / (2 * gradientDelta);
        }

        hyperParameterVertex.setAndCascade(hyperParameterValue);

        Map<Vertex, DoubleTensor> diffln = vertexUnderTest.dLogProbAtValue(hyperParameterVertex);

        double actualDiffLnDensity = diffln.get(hyperParameterVertex).scalar();

        assertEquals("Diff ln density problem at " + vertexUnderTest.getValue() + " hyper param value " + hyperParameterValue,
            diffLnDensityApproxExpected, actualDiffLnDensity, 0.1);
    }

    public static <V extends DoubleVertex & ProbabilisticDouble & Differentiable>
    void isTreatedAsConstantWhenObserved(V vertexUnderTest) {
        vertexUnderTest.observe(DoubleTensor.ones(vertexUnderTest.getValue().getShape()));
        assertTrue(Differentiator.reverseModeAutoDiff(vertexUnderTest, vertexUnderTest).asMap().isEmpty());
    }

    public static <V extends DoubleVertex & ProbabilisticDouble>
    void hasNoGradientWithRespectToItsValueWhenObserved(V vertexUnderTest) {
        DoubleTensor ones = DoubleTensor.ones(vertexUnderTest.getValue().getShape());
        vertexUnderTest.observe(ones);
        assertNull(vertexUnderTest.dLogProb(ones).get(vertexUnderTest));
    }

    public static void matchesKnownLogDensityOfVector(Probabilistic<DoubleTensor> vertexUnderTest, double[] vector, double expectedLogDensity) {

        double actualDensity = vertexUnderTest.logProb(DoubleTensor.create(vector, vector.length, 1));
        assertEquals(expectedLogDensity, actualDensity, 1e-5);
    }

    public static void matchesKnownLogDensityOfScalar(Probabilistic<DoubleTensor> vertexUnderTest, double scalar, double expectedLogDensity) {

        double actualDensity = vertexUnderTest.logProb(DoubleTensor.scalar(scalar));
        assertEquals(expectedLogDensity, actualDensity, 1e-5);
    }

    public static <V extends DoubleVertex & ProbabilisticDouble>
    void matchesKnownDerivativeLogDensityOfVector(double[] vector, Supplier<V> vertexUnderTestSupplier) {

        ImmutableList.Builder<V> scalarVertices = ImmutableList.builder();
        PartialDerivatives expectedPartialDerivatives = new PartialDerivatives(new HashMap<>());

        for (int i = 0; i < vector.length; i++) {

            V scalarVertex = vertexUnderTestSupplier.get();
            scalarVertices.add(scalarVertex);

            Map<VertexId, DoubleTensor> dlogPdfById = scalarVertex.dLogPdf(vector[i], scalarVertex)
                .entrySet().stream()
                .collect(toMap(
                    e -> e.getKey().getId(),
                    Map.Entry::getValue)
                );

            expectedPartialDerivatives = expectedPartialDerivatives.add(
                new PartialDerivatives(
                    dlogPdfById
                )
            );
        }

        V tensorVertex = vertexUnderTestSupplier.get();

        Map<Vertex, DoubleTensor> actualDerivatives = tensorVertex.dLogProb(
            DoubleTensor.create(vector, new long[]{vector.length, 1}),
            tensorVertex
        );

        HashSet<Vertex> hyperParameterVertices = new HashSet<>(actualDerivatives.keySet());
        hyperParameterVertices.remove(tensorVertex);

        for (Vertex vertex : hyperParameterVertices) {
            assertEquals(expectedPartialDerivatives.withRespectTo(vertex).sum(), actualDerivatives.get(vertex).sum(), 1e-5);
        }

        double expected = 0;
        for (V scalarVertex : scalarVertices.build()) {
            expected += expectedPartialDerivatives.withRespectTo(scalarVertex).scalar();
        }

        double actual = actualDerivatives.get(tensorVertex).sum();
        assertEquals(expected, actual, 1e-5);
    }

    public static <V extends DoubleVertex & ProbabilisticDouble>
    void sampleMethodMatchesLogProbMethodMultiVariate(V vertexUnderTest,
                                                      double from,
                                                      double to,
                                                      double bucketSize,
                                                      double maxError,
                                                      int sampleCount,
                                                      KeanuRandom random,
                                                      double bucketVolume,
                                                      boolean isVector) {
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

        long[] shape = isVector ? new long[]{1, 2} : new long[]{2, 1};

        for (Map.Entry<Pair<Double, Double>, Long> entry : sampleBucket.entrySet()) {
            double percentage = (double) entry.getValue() / sampleCount;
            if (percentage != 0) {
                double[] bucketCenter = new double[]{entry.getKey().getFirst(), entry.getKey().getSecond()};
                Nd4jDoubleTensor bucket = new Nd4jDoubleTensor(bucketCenter, shape);
                double densityAtBucketCenter = Math.exp(vertexUnderTest.logProb(bucket)) * bucketVolume;
                double actual = percentage;
                assertThat("Problem with logProb at " + bucketCenter, densityAtBucketCenter, closeTo(actual, maxError));
            }
        }

    }

}
