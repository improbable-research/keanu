package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import static java.util.stream.Collectors.counting;
import static java.util.stream.Collectors.groupingBy;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.number.IsCloseTo.closeTo;
import static org.junit.Assert.*;

public class ProbabilisticDoubleContract {

    /**
     * This method brute force verifies that a given vertex's sample method accurately reflects its logProb method.
     * This is done for a given range with a specified resolution (bucketSize). The error due to the approximate
     * nature of the brute force technique will be larger where the gradient of the logProb is large as well.
     *
     * @param vertexUnderTest
     * @param sampleCount
     * @param from
     * @param to
     * @param bucketSize
     */
    public static void sampleMethodMatchesLogProbMethod(Vertex<Double> vertexUnderTest,
                                                        long sampleCount,
                                                        double from,
                                                        double to,
                                                        double bucketSize,
                                                        double maxError,
                                                        KeanuRandom random) {
        double bucketCount = ((to - from) / bucketSize);

        if (bucketCount != (int) bucketCount) {
            throw new IllegalArgumentException("Range must be evenly divisible by bucketSize");
        }

        Map<Double, Long> histogram = Stream.generate(() -> vertexUnderTest.sample(random))
            .limit(sampleCount)
            .filter(value -> value >= from && value <= to)
            .collect(groupingBy(
                x -> bucketCenter(x, bucketSize, from),
                counting()
            ));

        for (Map.Entry<Double, Long> sampleBucket : histogram.entrySet()) {
            double percentage = (double) sampleBucket.getValue() / sampleCount;
            double bucketCenter = sampleBucket.getKey();

            double densityAtBucketCenter = Math.exp(vertexUnderTest.logProb(bucketCenter));
            double actual = percentage / bucketSize;

            assertThat("Problem with logProb at " + bucketCenter, densityAtBucketCenter, closeTo(actual, maxError));
        }
    }

    private static Double bucketCenter(Double x, double bucketSize, double from) {
        double bucketNumber = Math.floor((x - from) / bucketSize);
        return bucketNumber * bucketSize + bucketSize / 2 + from;
    }

    public static void samplingProducesRealisticMeanAndStandardDeviation(int numberOfSamples,
                                                                         Vertex<Double> vertexUnderTest,
                                                                         double expectedMean,
                                                                         double expectedStandardDeviation,
                                                                         double maxError,
                                                                         KeanuRandom random) {
        List<Double> samples = new ArrayList<>();

        for (int i = 0; i < numberOfSamples; i++) {
            double sample = vertexUnderTest.sample(random);
            samples.add(sample);
        }

        SummaryStatistics stats = new SummaryStatistics();
        samples.forEach(stats::addValue);

        double mean = stats.getMean();
        double sd = stats.getStandardDeviation();

        assertThat("Problem with mean", expectedMean, closeTo(mean, maxError));
        assertThat("Problem with standard deviation", sd, closeTo(expectedStandardDeviation, maxError));
    }

    public static void moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(double hyperParameterStartValue,
                                                                                          double hyperParameterEndValue,
                                                                                          double hyperParameterValueIncrement,
                                                                                          Vertex<Double> hyperParameterVertex,
                                                                                          Vertex<Double> vertexUnderTest,
                                                                                          double vertexStartValue,
                                                                                          double vertexEndValue,
                                                                                          double vertexValueIncrement,
                                                                                          double gradientDelta) {

        for (double value = vertexStartValue; value <= vertexEndValue; value += vertexValueIncrement) {
            vertexUnderTest.setAndCascade(value);
            testGradientAcrossMultipleHyperParameterValues(
                hyperParameterStartValue,
                hyperParameterEndValue,
                hyperParameterValueIncrement,
                hyperParameterVertex,
                value,
                vertexUnderTest,
                gradientDelta
            );
        }
    }

    public static void testGradientAcrossMultipleHyperParameterValues(double hyperParameterStartValue,
                                                                      double hyperParameterEndValue,
                                                                      double hyperParameterValueIncrement,
                                                                      Vertex<Double> hyperParameterVertex,
                                                                      double vertexValue,
                                                                      Vertex<Double> vertexUnderTest,
                                                                      double gradientDelta) {

        for (double parameterValue = hyperParameterStartValue; parameterValue <= hyperParameterEndValue; parameterValue += hyperParameterValueIncrement) {
            testGradientAtHyperParameterValue(
                parameterValue,
                hyperParameterVertex,
                vertexValue,
                vertexUnderTest,
                gradientDelta
            );
        }
    }

    public static void testGradientAtHyperParameterValue(double hyperParameterValue,
                                                         Vertex<Double> hyperParameterVertex,
                                                         double vertexValue,
                                                         Vertex<Double> vertexUnderTest,
                                                         double gradientDelta) {

        hyperParameterVertex.setAndCascade(hyperParameterValue - gradientDelta);
        double lnDensityA1 = vertexUnderTest.logProb(vertexValue);

        hyperParameterVertex.setAndCascade(hyperParameterValue + gradientDelta);
        double lnDensityA2 = vertexUnderTest.logProb(vertexValue);

        double diffLnDensityApproxExpected = (lnDensityA2 - lnDensityA1) / (2 * gradientDelta);

        hyperParameterVertex.setAndCascade(hyperParameterValue);

        Map<Long, DoubleTensor> diffln = vertexUnderTest.dLogProbAtValue();

        double actualDiffLnDensity = diffln.get(hyperParameterVertex.getId()).scalar();

        assertEquals("Diff ln density problem at " + vertexValue + " hyper param value " + hyperParameterValue,
            diffLnDensityApproxExpected, actualDiffLnDensity, 0.1);
    }

    public static void isTreatedAsConstantWhenObserved(DoubleVertex vertexUnderTest) {
        vertexUnderTest.observe(1.0);
        assertTrue(vertexUnderTest.getDualNumber().isOfConstant());
    }

    public static void hasNoGradientWithRespectToItsValueWhenObserved(DoubleVertex vertexUnderTest) {
        vertexUnderTest.observe(1.0);
        assertNull(vertexUnderTest.dLogProb(1.0).get(vertexUnderTest.getId()));
    }

}
