package io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic;

import com.google.common.collect.ImmutableList;
import com.google.common.primitives.Ints;
import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiable;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiator;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.LogProbGradientCalculator;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.LogProbGradients;
import org.apache.commons.math3.util.Pair;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
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

            double densityAtBucketCenter = Math.exp(vertexUnderTest.logProb(DoubleTensor.scalar(bucketCenter)));
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

            double densityAtBucketCenter = Math.exp(vertexUnderTest.logProb(DoubleTensor.scalar(bucketCenter)));
            double actual = percentage / bucketSize;
            assertThat("Problem with logProb at " + bucketCenter, densityAtBucketCenter, closeTo(actual, maxError));
        }
    }

    private static Double bucketCenter(Double x, double bucketSize, double from) {
        double bucketNumber = Math.floor((x - from) / bucketSize);
        return bucketNumber * bucketSize + bucketSize / 2 + from;
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

        for (DoubleTensor value = vertexStartValue; value.lessThanOrEqual(vertexEndValue).allTrue().scalar(); value.plusInPlace(vertexValueIncrement)) {
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

        DoubleTensor hyperInc = DoubleTensor.zeros(hyperParameterStartValue.getShape());
        int hyperParamLength = Ints.checkedCast(hyperInc.getLength());

        for (DoubleTensor parameterValue = hyperParameterStartValue; parameterValue.lessThan(hyperParameterEndValue).anyTrue().scalar(); parameterValue = parameterValue.plusInPlace(hyperInc)) {

            DoubleTensor gradientMask = DoubleTensor.zeros(hyperParameterStartValue.getShape());

            for (int i = 0; i < hyperParamLength; i++) {
                if (parameterValue.getValue(i) < hyperParameterEndValue.getValue(i)) {
                    hyperInc = DoubleTensor.zeros(hyperParameterStartValue.getShape());
                    hyperInc.setValue(hyperParameterValueIncrement, i);

                    gradientMask = DoubleTensor.zeros(hyperParameterStartValue.getShape());
                    gradientMask.setValue(1.0, i);
                    break;
                }
            }

            testGradientAtHyperParameterValue(
                parameterValue,
                hyperParameterVertex,
                vertexUnderTest,
                gradientDelta,
                gradientMask
            );
        }
    }

    public static void testGradientAtHyperParameterValue(DoubleTensor hyperParameterValue,
                                                         DoubleVertex hyperParameterVertex,
                                                         Probabilistic<?> vertexUnderTest,
                                                         double gradientDelta,
                                                         DoubleTensor gradientMask) {

        DoubleTensor delta = gradientMask.times(gradientDelta);
        hyperParameterVertex.setAndCascade(hyperParameterValue.minus(delta));
        double lnDensityA1 = vertexUnderTest.logProbAtValue();

        hyperParameterVertex.setAndCascade(hyperParameterValue.plus(delta));
        double lnDensityA2 = vertexUnderTest.logProbAtValue();

        double diffLnDensityApproxExpected = 0.0;
        if (lnDensityA1 != lnDensityA2) {
            diffLnDensityApproxExpected = (lnDensityA2 - lnDensityA1) / (2 * gradientDelta);
        }

        hyperParameterVertex.setAndCascade(hyperParameterValue);

        LogProbGradientCalculator gradient = new LogProbGradientCalculator(ImmutableList.of((Vertex) vertexUnderTest), ImmutableList.of(hyperParameterVertex));
        Map<VertexId, DoubleTensor> diffln = gradient.getJointLogProbGradientWrtLatents();

        double actualDiffLnDensity = diffln.get(hyperParameterVertex.getId()).times(gradientMask).sumNumber();

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
        LogProbGradients expectedPartialDerivatives = new LogProbGradients();

        for (int i = 0; i < vector.length; i++) {

            V scalarVertex = vertexUnderTestSupplier.get();
            scalarVertices.add(scalarVertex);

            Map<VertexId, DoubleTensor> dlogPdfById = scalarVertex.dLogPdf(vector[i], scalarVertex)
                .entrySet().stream()
                .collect(toMap(
                    e -> e.getKey().getId(),
                    Map.Entry::getValue)
                );

            expectedPartialDerivatives.add(dlogPdfById);
        }

        V tensorVertex = vertexUnderTestSupplier.get();

        Map<Vertex, DoubleTensor> actualDerivatives = tensorVertex.dLogProb(
            DoubleTensor.create(vector, new long[]{vector.length, 1}),
            tensorVertex
        );

        HashSet<Vertex> hyperParameterVertices = new HashSet<>(actualDerivatives.keySet());
        hyperParameterVertices.remove(tensorVertex);

        for (Vertex vertex : hyperParameterVertices) {
            assertEquals(expectedPartialDerivatives.getWithRespectTo(vertex.getId()).sumNumber(), actualDerivatives.get(vertex).sumNumber(), 1e-5);
        }

        double expected = 0;
        for (V scalarVertex : scalarVertices.build()) {
            expected += expectedPartialDerivatives.getWithRespectTo(scalarVertex.getId()).scalar();
        }

        double actual = actualDerivatives.get(tensorVertex).sumNumber();
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

        long[] shape = new long[]{2};

        for (Map.Entry<Pair<Double, Double>, Long> entry : sampleBucket.entrySet()) {
            double percentage = (double) entry.getValue() / sampleCount;
            if (percentage != 0) {
                double[] bucketCenter = new double[]{entry.getKey().getFirst(), entry.getKey().getSecond()};
                DoubleTensor bucket = DoubleTensor.create(bucketCenter, shape);
                double densityAtBucketCenter = Math.exp(vertexUnderTest.logProb(bucket)) * bucketVolume;
                double actual = percentage;
                assertThat("Problem with logProb at " + bucketCenter, densityAtBucketCenter, closeTo(actual, maxError));
            }
        }

    }

}
