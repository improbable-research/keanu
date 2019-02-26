package io.improbable.keanu.vertices.bool.probabilistic;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.Keanu;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizer;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.LogProbGraphContract;
import io.improbable.keanu.vertices.LogProbGraphValueFeeder;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.LogProbGradientCalculator;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Rule;
import org.junit.Test;

import java.util.Map;

import static io.improbable.keanu.tensor.TensorMatchers.valuesWithinEpsilonAndShapesMatch;
import static io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDoubleTensorContract.testGradientAcrossMultipleHyperParameterValues;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class BernoulliVertexTest {

    @Rule
    public DeterministicRule rule = new DeterministicRule();

    @Test
    public void doesTensorSample() {
        long[] expectedShape = new long[]{1, 100};
        BernoulliVertex bernoulliVertex = new BernoulliVertex(expectedShape, 0.25);
        BooleanTensor samples = bernoulliVertex.sample();
        assertArrayEquals(expectedShape, samples.getShape());
    }

    @Test
    public void doesExpectedLogProbOnTensor() {
        double probTrue = 0.25;
        BernoulliVertex bernoulliVertex = new BernoulliVertex(new long[]{1, 2}, probTrue);
        double actualLogPmf = bernoulliVertex.logPmf(BooleanTensor.create(true, false));
        double expectedLogPmf = Math.log(probTrue) + Math.log(1 - probTrue);
        assertEquals(expectedLogPmf, actualLogPmf, 1e-10);
    }

    @Test
    public void doesExpectedLogProbGraphOnTensor() {
        DoubleVertex probTrue = ConstantVertex.of(0.25, 0.25);
        BernoulliVertex bernoulliVertex = new BernoulliVertex(probTrue);
        LogProbGraph logProbGraph = bernoulliVertex.logProbGraph();
        LogProbGraphValueFeeder.feedValue(logProbGraph, probTrue, probTrue.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, bernoulliVertex, BooleanTensor.create(true, false));
        double expectedDensity = Math.log(0.25) + Math.log(0.75);
        LogProbGraphContract.matchesKnownLogDensity(logProbGraph, expectedDensity);
    }

    @Test
    public void logProbClampsProbTrueTo1() {
        double probTrue = 2.;
        BernoulliVertex bernoulliVertex = new BernoulliVertex(probTrue);
        double actualLogPmf = bernoulliVertex.logPmf(BooleanTensor.create(true));
        assertEquals(0., actualLogPmf, 1e-10);
    }

    @Test
    public void logProbGraphClampsProbTrueTo1() {
        DoubleVertex probTrue = ConstantVertex.of(2.);
        BernoulliVertex bernoulliVertex = new BernoulliVertex(probTrue);
        LogProbGraph logProbGraph = bernoulliVertex.logProbGraph();
        LogProbGraphValueFeeder.feedValue(logProbGraph, probTrue, probTrue.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, bernoulliVertex, BooleanTensor.scalar(true));
        LogProbGraphContract.matchesKnownLogDensity(logProbGraph, 0.);
    }

    @Test
    public void logProbClampsProbTrueTo0() {
        double probTrue = -1.;
        BernoulliVertex bernoulliVertex = new BernoulliVertex(probTrue);
        double actualLogPmf = bernoulliVertex.logPmf(BooleanTensor.create(false));
        assertEquals(0., actualLogPmf, 1e-10);
    }

    @Test
    public void logProbGraphClampsProbTrueTo0() {
        DoubleVertex probTrue = ConstantVertex.of(-1.);
        BernoulliVertex bernoulliVertex = new BernoulliVertex(probTrue);
        LogProbGraph logProbGraph = bernoulliVertex.logProbGraph();
        LogProbGraphValueFeeder.feedValue(logProbGraph, probTrue, probTrue.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, bernoulliVertex, BooleanTensor.scalar(false));
        LogProbGraphContract.matchesKnownLogDensity(logProbGraph, 0.);
    }

    @Test
    public void doesCalculateDiffLogProbWithRespectToHyperParamHandCalculated() {

        DoubleVertex A = new GaussianVertex(new long[]{1, 2}, 0, 1);
        A.setValue(new double[]{0.25, 0.6});
        DoubleVertex B = new GaussianVertex(new long[]{1, 2}, 0, 1);
        B.setValue(new double[]{0.5, 0.2});
        DoubleVertex C = A.times(B);
        BernoulliVertex D = new BernoulliVertex(C);

        D.observe(BooleanTensor.create(true, false));

        LogProbGradientCalculator logProbGradientCalculator = new LogProbGradientCalculator(ImmutableList.of(D), ImmutableList.of(A, B));

        Map<VertexId, DoubleTensor> dLogPmf = logProbGradientCalculator.getJointLogProbGradientWrtLatents();

        DoubleTensor expectedWrtA = DoubleTensor.create((1.0 / 0.125) * 0.5, (-1.0 / 0.88) * 0.2);
        DoubleTensor expectedWrtB = DoubleTensor.create((1.0 / 0.125) * 0.25, (-1.0 / 0.88) * 0.6);

        assertEquals(expectedWrtA, dLogPmf.get(A.getId()));
        assertEquals(expectedWrtB, dLogPmf.get(B.getId()));
    }

    @Test
    public void doesCalculateDiffLogProbWithRespectToHyperParamFiniteElement() {

        DoubleVertex A = new GaussianVertex(0, 1);
        DoubleTensor startA = DoubleTensor.scalar(0.1);
        DoubleTensor endA = DoubleTensor.scalar(0.9);
        A.setAndCascade(startA);

        BernoulliVertex D = new BernoulliVertex(A);
        D.observe(true);

        double increment = 0.1;
        double gradientDelta = 1e-5;

        testGradientAcrossMultipleHyperParameterValues(
            startA,
            endA,
            increment,
            A,
            D,
            gradientDelta
        );

        D.observe(false);

        testGradientAcrossMultipleHyperParameterValues(
            startA,
            endA,
            increment,
            A,
            D,
            gradientDelta
        );
    }

    @Test
    public void doesCalculateDiffLogProbWithRespectToRank3HyperParam() {

        long[] shape = new long[]{2, 2, 2};
        DoubleVertex A = new GaussianVertex(shape, 0, 1);
        DoubleTensor AValue = DoubleTensor.create(new double[]{
            0.1, 0.2,
            0.3, 0.4,
            5, 0.3,
            0.2, 0.8
        }, shape);

        A.setValue(AValue);

        DoubleVertex B = new GaussianVertex(shape, 0, 1);
        DoubleTensor BValue = DoubleTensor.create(new double[]{
            0.55, 0.65,
            0.45, 0.25,
            0.8, -0.4,
            0.5, 0.3,
        }, shape);

        B.setValue(BValue);

        DoubleVertex C = A.times(B);
        BernoulliVertex D = new BernoulliVertex(C);

        BooleanTensor atValue = BooleanTensor.create(new boolean[]{
            true, false,
            false, true,
            false, false,
            true, true
        }, shape);

        D.observe(atValue);

        LogProbGradientCalculator logProbGradientCalculator = new LogProbGradientCalculator(ImmutableList.of(D), ImmutableList.of(A, B));

        Map<VertexId, DoubleTensor> dLogPmf = logProbGradientCalculator.getJointLogProbGradientWrtLatents();

        DoubleTensor expectedWrtA = atValue.doubleWhere(
            AValue.reciprocal(),
            BValue.div(AValue.times(BValue).minus(1.0))
        );

        expectedWrtA = expectedWrtA.setWithMaskInPlace(
            AValue.times(BValue).getGreaterThanMask(DoubleTensor.ONE_SCALAR),
            0.0
        );

        expectedWrtA = expectedWrtA.setWithMaskInPlace(
            AValue.times(BValue).getLessThanOrEqualToMask(DoubleTensor.ZERO_SCALAR),
            0.0
        );

        DoubleTensor expectedWrtB = atValue.doubleWhere(
            BValue.reciprocal(),
            AValue.div(AValue.times(BValue).minus(1.0))
        );

        expectedWrtB = expectedWrtB.setWithMaskInPlace(
            AValue.times(BValue).getGreaterThanMask(DoubleTensor.ONE_SCALAR),
            0.0
        );

        expectedWrtB = expectedWrtB.setWithMaskInPlace(
            AValue.times(BValue).getLessThanOrEqualToMask(DoubleTensor.ZERO_SCALAR),
            0.0
        );

        assertThat(expectedWrtA, valuesWithinEpsilonAndShapesMatch(dLogPmf.get(A.getId()), 1e-8));
        assertThat(expectedWrtB, valuesWithinEpsilonAndShapesMatch(dLogPmf.get(B.getId()), 1e-8));
    }

    @Test
    public void canUseWithGradientOptimizer() {

        int n = 1;
        double min = 0;
        double max = 1;
        DoubleVertex A = new UniformVertex(new long[]{1, n}, min, max);

        BernoulliVertex observedVertex = new BernoulliVertex(A.sigmoid());
        BooleanTensor observation = observedVertex.sample();
        observedVertex.observe(observation);

        BayesianNetwork network = new BayesianNetwork(observedVertex.getConnectedGraph());
        GradientOptimizer optimizer = Keanu.Optimizer.Gradient.of(network);

        optimizer.maxAPosteriori();

        DoubleTensor expected = observation.doubleWhere(
            DoubleTensor.scalar(max),
            DoubleTensor.scalar(min)
        );

        DoubleTensor actual = A.getValue();

        assertArrayEquals(expected.asFlatDoubleArray(), actual.asFlatDoubleArray(), 0.1);
    }

}
