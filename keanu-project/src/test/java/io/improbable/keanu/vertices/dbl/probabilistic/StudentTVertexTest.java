package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.vertices.dbl.KeanuRandom;
import org.apache.commons.math3.distribution.TDistribution;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

import static java.lang.Math.pow;
import static junit.framework.TestCase.assertEquals;

public class StudentTVertexTest {
    private static final double DELTA = 0.0001;
    private static final int[] TEST_VALUES_OF_V = new int[]{
        1, 2, 3
    };
    private final Logger log = LoggerFactory.getLogger(StudentTVertexTest.class);
    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void testSamplesMatchExpectedMeanAndVariance() {
        int N = 5_000;

        double sampleDelta = 0.1;

        int v = 3;

        StudentTVertex studentT = new StudentTVertex(new int[]{N, 1}, v);

        List<Double> samples = studentT.sample(random).asFlatList();

        testSampleMeanAndStdDeviation(v, 0.0, Math.sqrt(v / (v - 2)), samples, sampleDelta);
    }

    @Test
    public void testLogPdfTest() {
        for (int testValueForV : TEST_VALUES_OF_V) {
            testLogPdfAtGivenDegreesOfFreedom(testValueForV);
        }
    }

    /**
     * Test the differential of the log of the StudentT Probability Density Function
     */
    @Test
    public void dLogPdfTest() {
        for (int testValueForV : TEST_VALUES_OF_V) {
            testDLogPdfAtGivenDegreesOfFreedom(testValueForV);
        }
    }

    private void testSampleMeanAndStdDeviation(int v, double expectedMean, double expectedSd, List<Double> samples, double delta) {
        SummaryStatistics stats = new SummaryStatistics();
        samples.forEach(stats::addValue);

        double mean = stats.getMean();
        double sd = stats.getStandardDeviation();
        log.trace("Degrees of freedom: " + v);
        log.trace("Mean: " + mean);
        log.trace("Standard deviation: " + sd);
        Assert.assertEquals(expectedMean, mean, delta);
        Assert.assertEquals(expectedSd, sd, delta);
    }

    private void testLogPdfAtGivenDegreesOfFreedom(int v) {
        TDistribution apache = new TDistribution(v);
        StudentTVertex studentT = new StudentTVertex(v);

        for (double t = -4.5; t <= 4.5; t += 0.5) {
            double expected = apache.logDensity(t);
            double actual = studentT.logPdf(t);
            assertEquals(expected, actual, DELTA);
        }
    }

    private void testDLogPdfAtGivenDegreesOfFreedom(int v) {
        StudentTVertex studentT = new StudentTVertex(v);

        for (double t = -4.5; t <= 4.5; t += 0.5) {
            double expected;
            double actual = studentT.dLogPdf(t).get(studentT.getId()).scalar();
            switch (v) {
                case 1:
                    expected = (-2 * t) / (pow(t, 2) + 1.);
                    break;
                case 2:
                    expected = (-3 * t) / (pow(t, 2) + 2.);
                    break;
                case 3:
                    expected = (-4 * t) / (pow(t, 2) + 3.);
                    break;
                default:
                    expected = 0.;
            }
            assertEquals(expected, actual, DELTA);
        }
    }
}
