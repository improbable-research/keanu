package io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.Keanu;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizer;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.LogProbGraphContract;
import io.improbable.keanu.vertices.LogProbGraphValueFeeder;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.IntegerVertex;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.distribution.TDistribution;
import org.junit.Rule;
import org.junit.Test;

import static io.improbable.keanu.tensor.TensorMatchers.valuesWithinEpsilonAndShapesMatch;
import static java.lang.Math.pow;
import static junit.framework.TestCase.assertEquals;
import static org.junit.Assert.assertThat;

@Slf4j
public class StudentTVertexTest {
    private static final double DELTA = 0.0001;
    private static final int[] TEST_VALUES_OF_V = new int[]{
        1, 2, 3
    };

    @Rule
    public DeterministicRule deterministicRule = new DeterministicRule();

    @Test
    public void testSamplesMatchExpectedMeanAndVariance() {
        int N = 5_000;

        IntegerTensor v = IntegerTensor.create(3, 4);

        StudentTVertex studentT = new StudentTVertex(new long[]{N, 2, 2}, ConstantVertex.of(v));

        DoubleTensor sample = studentT.sample();

        DoubleTensor vDbl = v.broadcast(2, 2).toDouble();
        DoubleTensor expectedStd = vDbl.div(vDbl.minus(2)).sqrt();

        //Calculate variance of sample along dimension 0
        DoubleTensor stdActual = sample.minus(sample.mean(0)).pow(2).sum(0).div(N).sqrt();

        assertThat(stdActual, valuesWithinEpsilonAndShapesMatch(expectedStd, 1e-1));
    }

    @Test
    public void testLogPdfTest() {
        for (int testValueForV : TEST_VALUES_OF_V) {
            testLogPdfAtGivenDegreesOfFreedom(testValueForV);
        }
    }

    @Test
    public void logProbGraphMatchesKnownLogDensityOfScalar() {
        IntegerVertex v = ConstantVertex.of(1);
        StudentTVertex studentT = new StudentTVertex(v);
        LogProbGraph logProbGraph = studentT.logProbGraph();

        LogProbGraphValueFeeder.feedValue(logProbGraph, v, v.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, studentT, DoubleTensor.scalar(-4.5));

        TDistribution distribution = new TDistribution(1);
        double expectedDensity = distribution.logDensity(-4.5);

        LogProbGraphContract.matchesKnownLogDensity(logProbGraph, expectedDensity);
    }

    @Test
    public void logProbGraphMatchesKnownLogDensityOfVector() {
        IntegerVertex v = ConstantVertex.of(1, 1);
        StudentTVertex studentT = new StudentTVertex(v);
        LogProbGraph logProbGraph = studentT.logProbGraph();

        LogProbGraphValueFeeder.feedValue(logProbGraph, v, v.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, studentT, DoubleTensor.create(-4.5, 4.5));

        TDistribution distribution = new TDistribution(1);
        double expectedDensity = distribution.logDensity(-4.5) + distribution.logDensity(4.5);

        LogProbGraphContract.matchesKnownLogDensity(logProbGraph, expectedDensity);
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
            double actual = studentT.dLogPdf(t, studentT).get(studentT).scalar();
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

    @Test
    public void calcMAP() {
        IntegerTensor v = IntegerTensor.create(4);
        StudentTVertex studentTVertex = new StudentTVertex(ConstantVertex.of(v));
        studentTVertex.setAndCascade(DoubleTensor.create(9.0));
        GradientOptimizer optimizer = Keanu.Optimizer.Gradient.ofConnectedGraph(studentTVertex);

        optimizer.maxAPosteriori();
        assertThat(studentTVertex.getValue(), valuesWithinEpsilonAndShapesMatch(DoubleTensor.create(0), 1e-3));
    }

    @Test
    public void calcBatchMAP() {
        IntegerTensor v = IntegerTensor.create(4, 3);
        StudentTVertex studentTVertex = new StudentTVertex(new long[]{2, 2}, ConstantVertex.of(v));
        studentTVertex.setAndCascade(DoubleTensor.create(9, 5, 6, 7).reshape(2, 2));
        GradientOptimizer optimizer = Keanu.Optimizer.Gradient.ofConnectedGraph(studentTVertex);

        optimizer.maxAPosteriori();
        assertThat(studentTVertex.getValue(), valuesWithinEpsilonAndShapesMatch(DoubleTensor.create(0, 0, 0, 0).reshape(2, 2), 1e-3));
    }
}
