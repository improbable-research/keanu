package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.vertices.Vertex;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import static java.util.stream.Collectors.counting;
import static java.util.stream.Collectors.groupingBy;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.number.IsCloseTo.closeTo;
import static org.junit.Assert.assertEquals;

public class ProbabilisticDoubleContract {

    /**
     * This method brute force verifies that a given vertex's sample method accurately reflects its density method.
     * This is done for a given range with a specified resolution (bucketSize). The error due to the approximate
     * nature of the brute force technique will be larger where the gradient of the density is large as well.
     *
     * @param vertexUnderTest
     * @param sampleCount
     * @param from
     * @param to
     * @param bucketSize
     */
    public static void sampleMethodMatchesDensityMethod(Vertex<Double> vertexUnderTest,
                                                        long sampleCount,
                                                        double from,
                                                        double to,
                                                        double bucketSize,
                                                        double maxError) {
        double bucketCount = ((to - from) / bucketSize);

        if (bucketCount != (int) bucketCount) {
            throw new IllegalArgumentException("Range must be evenly divisible by bucketSize");
        }

        Map<Double, Long> histogram = Stream.generate(vertexUnderTest::sample)
                .limit(sampleCount)
                .filter(value -> value >= from && value <= to)
                .collect(groupingBy(
                        x -> bucketCenter(x, bucketSize, from),
                        counting()
                ));

        for (Map.Entry<Double, Long> sampleBucket : histogram.entrySet()) {
            double percentage = (double) sampleBucket.getValue() / sampleCount;
            double bucketCenter = sampleBucket.getKey();

            double densityAtBucketCenter = vertexUnderTest.density(bucketCenter);
            double actual = percentage / bucketSize;

            assertThat("Problem with density at " + bucketCenter, densityAtBucketCenter, closeTo(actual, maxError));
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
                                                                         double maxError) {
        List<Double> samples = new ArrayList<>();

        for (int i = 0; i < numberOfSamples; i++) {
            double sample = vertexUnderTest.sample();
            samples.add(sample);
        }

        SummaryStatistics stats = new SummaryStatistics();
        samples.forEach(stats::addValue);

        double mean = stats.getMean();
        double sd = stats.getStandardDeviation();

        assertThat("Problem with mean", expectedMean, closeTo(mean, maxError));
        assertThat("Problem with standard deviation", expectedStandardDeviation, closeTo(sd, maxError));
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
            testGradientAcrossMultipleHyperParameterValues(hyperParameterStartValue, hyperParameterEndValue, hyperParameterValueIncrement, hyperParameterVertex, value, vertexUnderTest, gradientDelta);
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
            testGradientAtHyperParameterValue(parameterValue, hyperParameterVertex, vertexValue, vertexUnderTest, gradientDelta);
        }
    }

    public static void testGradientAtHyperParameterValue(double hyperParameterValue, Vertex<Double> hyperParameterVertex, double vertexValue, Vertex<Double> vertexUnderTest, double gradientDelta) {
        hyperParameterVertex.setAndCascade(hyperParameterValue - gradientDelta);
        double densityA1 = vertexUnderTest.density(vertexValue);
        double lnDensityA1 = vertexUnderTest.logDensity(vertexValue);

        hyperParameterVertex.setAndCascade(hyperParameterValue + gradientDelta);
        double densityA2 = vertexUnderTest.density(vertexValue);
        double lnDensityA2 = vertexUnderTest.logDensity(vertexValue);

        double diffDensityApproxExpected = (densityA2 - densityA1) / (2 * gradientDelta);
        double diffLnDensityApproxExpected = (lnDensityA2 - lnDensityA1) / (2 * gradientDelta);

        hyperParameterVertex.setAndCascade(hyperParameterValue);

        Map<String, Double> diff = vertexUnderTest.dDensityAtValue();
        Map<String, Double> diffln = vertexUnderTest.dlnDensityAtValue();

        double actualDiffDensity = diff.get(hyperParameterVertex.getId());
        double actualDiffLnDensity = diffln.get(hyperParameterVertex.getId());

        assertThat("Diff density problem at " + vertexValue + " hyper param value " + hyperParameterValue,
                diffDensityApproxExpected, closeTo(actualDiffDensity, 0.1));

        assertEquals("Diff ln density problem at " + vertexValue + " hyper param value " + hyperParameterValue,
                diffLnDensityApproxExpected, actualDiffLnDensity, 0.1);
    }

    public static void diffLnDensityIsSameAsLogOfDiffDensity(Vertex<Double> vertexUnderTest,
                                                             double value,
                                                             double maxError) {
        vertexUnderTest.setAndCascade(value);

        Map<String, Double> dP = vertexUnderTest.dDensityAtValue();
        Map<String, Double> dlnP = vertexUnderTest.dlnDensityAtValue();

        final double density = vertexUnderTest.densityAtValue();
        for (String vertexId : dP.keySet()) {
            dP.put(vertexId, dP.get(vertexId) / density);
        }

        assertEquals(dP.get(vertexUnderTest.getId()), dlnP.get(vertexUnderTest.getId()), maxError);
    }

}
