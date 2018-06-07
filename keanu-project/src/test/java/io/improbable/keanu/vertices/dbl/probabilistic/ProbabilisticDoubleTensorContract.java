package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;

import java.util.*;
import java.util.function.Supplier;

import static java.util.stream.Collectors.counting;
import static java.util.stream.Collectors.groupingBy;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.number.IsCloseTo.closeTo;
import static org.junit.Assert.*;

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
    public static void sampleMethodMatchesLogProbMethod(Vertex<DoubleTensor> vertexUnderTest,
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

    private static Double bucketCenter(Double x, double bucketSize, double from) {
        double bucketNumber = Math.floor((x - from) / bucketSize);
        return bucketNumber * bucketSize + bucketSize / 2 + from;
    }

    public static void samplingProducesRealisticMeanAndStandardDeviation(int numberOfSamples,
                                                                         Vertex<DoubleTensor> vertexUnderTest,
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

    public static void moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(DoubleTensor hyperParameterStartValue,
                                                                                          DoubleTensor hyperParameterEndValue,
                                                                                          double hyperParameterValueIncrement,
                                                                                          Vertex<DoubleTensor> hyperParameterVertex,
                                                                                          Vertex<DoubleTensor> vertexUnderTest,
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
                value,
                vertexUnderTest,
                gradientDelta
            );
        }
    }

    public static void testGradientAcrossMultipleHyperParameterValues(DoubleTensor hyperParameterStartValue,
                                                                      DoubleTensor hyperParameterEndValue,
                                                                      double hyperParameterValueIncrement,
                                                                      Vertex<DoubleTensor> hyperParameterVertex,
                                                                      DoubleTensor vertexValue,
                                                                      Vertex<DoubleTensor> vertexUnderTest,
                                                                      double gradientDelta) {

        for (DoubleTensor parameterValue = hyperParameterStartValue; parameterValue.scalar() <= hyperParameterEndValue.scalar(); parameterValue.plusInPlace(hyperParameterValueIncrement)) {
            testGradientAtHyperParameterValue(
                parameterValue,
                hyperParameterVertex,
                vertexValue,
                vertexUnderTest,
                gradientDelta
            );
        }
    }

    public static void testGradientAtHyperParameterValue(DoubleTensor hyperParameterValue,
                                                         Vertex<DoubleTensor> hyperParameterVertex,
                                                         DoubleTensor vertexValue,
                                                         Vertex<DoubleTensor> vertexUnderTest,
                                                         double gradientDelta) {

        hyperParameterVertex.setAndCascade(hyperParameterValue.minus(gradientDelta));
        double lnDensityA1 = vertexUnderTest.logProb(vertexValue);

        hyperParameterVertex.setAndCascade(hyperParameterValue.plus(gradientDelta));
        double lnDensityA2 = vertexUnderTest.logProb(vertexValue);

        double diffLnDensityApproxExpected = (lnDensityA2 - lnDensityA1) / (2 * gradientDelta);

        hyperParameterVertex.setAndCascade(hyperParameterValue);

        Map<Long, DoubleTensor> diffln = vertexUnderTest.dLogProbAtValue();

        double actualDiffLnDensity = diffln.get(hyperParameterVertex.getId()).scalar();

        assertEquals("Diff ln density problem at " + vertexValue + " hyper param value " + hyperParameterValue,
            diffLnDensityApproxExpected, actualDiffLnDensity, 0.1);
    }

    public static void isTreatedAsConstantWhenObserved(DoubleVertex vertexUnderTest) {
        vertexUnderTest.observe(DoubleTensor.ones(vertexUnderTest.getValue().getShape()));
        assertTrue(vertexUnderTest.getDualNumber().isOfConstant());
    }

    public static void hasNoGradientWithRespectToItsValueWhenObserved(DoubleVertex vertexUnderTest) {
        DoubleTensor ones = DoubleTensor.ones(vertexUnderTest.getValue().getShape());
        vertexUnderTest.observe(ones);
        assertNull(vertexUnderTest.dLogProb(ones).get(vertexUnderTest.getId()));
    }

    public static void matchesKnownLogDensityOfVector(DoubleVertex vertexUnderTest, double[] vector, double expectedLogDensity) {

        double actualDensity = vertexUnderTest.logPdf(DoubleTensor.create(vector, new int[]{vector.length, 1}));
        assertEquals(expectedLogDensity, actualDensity, 1e-5);
    }

    public static void matchesKnownLogDensityOfScalar(DoubleVertex vertexUnderTest, double scalar, double expectedLogDensity) {

        double actualDensity = vertexUnderTest.logPdf(Nd4jDoubleTensor.scalar(scalar));
        assertEquals(expectedLogDensity, actualDensity, 1e-5);
    }

    public static void matchesKnownDerivativeLogDensityOfVector(double[] vector, Supplier<DoubleVertex> vertexUnderTestSupplier) {

        DoubleVertex[] scalarVertices = new DoubleVertex[vector.length];
        PartialDerivatives tensorPartialDerivatives = new PartialDerivatives(new HashMap<>());

        for (int i = 0; i < vector.length; i++) {

            scalarVertices[i] = vertexUnderTestSupplier.get();

            tensorPartialDerivatives = tensorPartialDerivatives.add(
                new PartialDerivatives(
                    scalarVertices[i].dLogPdf(vector[i])
                )
            );
        }

        DoubleVertex tensorVertex = vertexUnderTestSupplier.get();

        Map<Long, DoubleTensor> actualDerivatives = tensorVertex.dLogPdf(
            DoubleTensor.create(vector, new int[]{vector.length, 1})
        );

        HashSet<Long> hyperParameterVertices = new HashSet<>(actualDerivatives.keySet());
        hyperParameterVertices.remove(tensorVertex.getId());

        for (Long id : hyperParameterVertices) {
            assertEquals(tensorPartialDerivatives.withRespectTo(id).sum(), actualDerivatives.get(id).sum(), 1e-5);
        }

        for (int i = 0; i < vector.length; i++) {
            assertEquals(tensorPartialDerivatives.withRespectTo(scalarVertices[i]).scalar(), actualDerivatives.get(tensorVertex.getId()).getValue(i), 1e-5);
        }
    }

}
