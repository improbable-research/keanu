package io.improbable.keanu.vertices.bool.probabilistic;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import static io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDoubleTensorContract.testGradientAcrossMultipleHyperParameterValues;

import java.util.Map;

import org.junit.Rule;
import org.junit.Test;

import com.google.common.collect.ImmutableSet;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizer;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;

public class BernoulliVertexTest {

    @Rule
    public DeterministicRule rule = new DeterministicRule();

    @Test
    public void doesTensorSample() {
        int[] expectedShape = new int[]{1, 100};
        BernoulliVertex bernoulliVertex = new BernoulliVertex(expectedShape, 0.25);
        BooleanTensor samples = bernoulliVertex.sample();
        assertArrayEquals(expectedShape, samples.getShape());
    }

    @Test
    public void doesExpectedLogProbOnTensor() {
        double probTrue = 0.25;
        BernoulliVertex bernoulliVertex = new BernoulliVertex(new int[]{1, 2}, probTrue);
        double actualLogPmf = bernoulliVertex.logPmf(BooleanTensor.create(new boolean[]{true, false}));
        double expectedLogPmf = Math.log(probTrue) + Math.log(1 - probTrue);
        assertEquals(expectedLogPmf, actualLogPmf, 1e-10);
    }

    @Test
    public void doesCalculateDiffLogProbWithRespectToHyperParamHandCalculated() {

        DoubleVertex A = new GaussianVertex(new int[]{1, 2}, 0, 1);
        A.setValue(new double[]{0.25, 0.6});
        DoubleVertex B = new GaussianVertex(new int[]{1, 2}, 0, 1);
        B.setValue(new double[]{0.5, 0.2});
        DoubleVertex C = A.times(B);
        BernoulliVertex D = new BernoulliVertex(C);

        Map<Vertex, DoubleTensor> dLogPmf = D.dLogPmf(BooleanTensor.create(new boolean[]{true, false}), ImmutableSet.of(A, B));

        DoubleTensor expectedWrtA = DoubleTensor.create(new double[]{(1.0 / 0.125) * 0.5, (-1.0 / 0.88) * 0.2});
        DoubleTensor expectedWrtB = DoubleTensor.create(new double[]{(1.0 / 0.125) * 0.25, (-1.0 / 0.88) * 0.6});

        assertEquals(expectedWrtA, dLogPmf.get(A));
        assertEquals(expectedWrtB, dLogPmf.get(B));
    }

    @Test
    public void doesCalculateDiffLogProbWithRespectToHyperParamFiniteElement() {

        DoubleVertex A = new GaussianVertex(0, 1);
        DoubleTensor startA = DoubleTensor.scalar(-5);
        DoubleTensor endA = DoubleTensor.scalar(5);
        A.setAndCascade(startA);

        BernoulliVertex D = new BernoulliVertex(A.sigmoid());
        D.observe(true);

        double increment = 0.15;
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

        int[] shape = new int[]{2, 2, 2};
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

        Map<Vertex, DoubleTensor> dLogPmf = D.dLogPmf(atValue, ImmutableSet.of(A, B));

        DoubleTensor expectedWrtA = atValue.setDoubleIf(
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

        DoubleTensor expectedWrtB = atValue.setDoubleIf(
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

        assertEquals(expectedWrtA, dLogPmf.get(A));
        assertEquals(expectedWrtB, dLogPmf.get(B));
    }

    @Test
    public void canUseWithGradientOptimizer() {

        int n = 1;
        double min = 0;
        double max = 1;
        DoubleVertex A = new UniformVertex(new int[]{1, n}, min, max);

        BernoulliVertex observedVertex = new BernoulliVertex(A.sigmoid());
        BooleanTensor observation = observedVertex.sample();
        observedVertex.observe(observation);

        BayesianNetwork network = new BayesianNetwork(observedVertex.getConnectedGraph());
        GradientOptimizer optimizer = GradientOptimizer.of(network);

        optimizer.maxAPosteriori();

        DoubleTensor expected = observation.setDoubleIf(
            DoubleTensor.scalar(max),
            DoubleTensor.scalar(min)
        );

        DoubleTensor actual = A.getValue();

        assertArrayEquals(expected.asFlatDoubleArray(), actual.asFlatDoubleArray(), 0.1);
    }

}
