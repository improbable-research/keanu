package io.improbable.keanu.vertices.bool.probabilistic;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.algorithms.variational.GradientOptimizer;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Rule;
import org.junit.Test;

import java.util.Arrays;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class FlipVertexTest {

    @Rule
    public DeterministicRule rule = new DeterministicRule();

    @Test
    public void doesTensorSample() {
        int[] expectedShape = new int[]{1, 100};
        Flip flip = new Flip(expectedShape, 0.25);
        BooleanTensor samples = flip.sample();
        assertArrayEquals(expectedShape, samples.getShape());
    }

    @Test
    public void doesExpectedLogProbOnTensor() {
        double probTrue = 0.25;
        Flip flip = new Flip(new int[]{1, 2}, probTrue);
        double actualLogPmf = flip.logPmf(BooleanTensor.create(new boolean[]{true, false}));
        double expectedLogPmf = Math.log(probTrue) + Math.log(1 - probTrue);
        assertEquals(expectedLogPmf, actualLogPmf, 1e-10);
    }

    @Test
    public void doesCalculateDiffLogProbWithRespectToHyperParam() {

        DoubleVertex A = new GaussianVertex(new int[]{1, 2}, 0, 1);
        A.setValue(new double[]{0.25, 0.6});
        DoubleVertex B = new GaussianVertex(new int[]{1, 2}, 0, 1);
        B.setValue(new double[]{0.5, 0.2});
        DoubleVertex C = A.times(B);
        Flip D = new Flip(C);

        Map<Long, DoubleTensor> dLogPmf = D.dLogPmf(BooleanTensor.create(new boolean[]{true, false}));

        DoubleTensor expectedWrtA = DoubleTensor.create(new double[]{0.5, -0.2});
        DoubleTensor expectedWrtB = DoubleTensor.create(new double[]{0.25, -0.6});

        assertEquals(expectedWrtA, dLogPmf.get(A.getId()));
        assertEquals(expectedWrtB, dLogPmf.get(B.getId()));
    }

    @Test
    public void doesCalculateDiffLogProbWithRespectToRank3HyperParam() {

        int[] shape = new int[]{2, 2, 2};
        DoubleVertex A = new GaussianVertex(shape, 0, 1);
        A.setValue(DoubleTensor.create(new double[]{
            0.1, 0.2,
            0.3, 0.4,
            0.5, 0.3,
            0.2, 0.8
        }, shape));

        DoubleVertex B = new GaussianVertex(shape, 0, 1);
        B.setValue(DoubleTensor.create(new double[]{
            0.55, 0.65,
            0.45, 0.25,
            0.3, 0.4,
            0.5, 0.3,
        }, shape));
        DoubleVertex C = A.times(B);
        Flip D = new Flip(C);

        Map<Long, DoubleTensor> dLogPmf = D.dLogPmf(BooleanTensor.create(new boolean[]{
            true, false,
            false, true,
            false, false,
            true, true
        }, shape));

        DoubleTensor expectedWrtA = DoubleTensor.create(new double[]{
            0.55, -0.65,
            -0.45, 0.25,
            -0.3, -0.4,
            0.5, 0.3,
        }, shape);

        DoubleTensor expectedWrtB = DoubleTensor.create(new double[]{
            0.1, -0.2,
            -0.3, 0.4,
            -0.5, -0.3,
            0.2, 0.8
        }, shape);

        assertEquals(expectedWrtA, dLogPmf.get(A.getId()));
        assertEquals(expectedWrtB, dLogPmf.get(B.getId()));
    }

    @Test
    public void canUseWithGradientOptimizer() {

        DoubleVertex A = new UniformVertex(new int[]{1, 2}, -4.0, 8.0);
        A.setValue(DoubleTensor.create(new double[]{-0.5, 0.5}));
        DoubleVertex B = A.sigmoid();
        Flip flip = new Flip(B);
        flip.observe(new boolean[]{true, false});

        GradientOptimizer optimizer = new GradientOptimizer(new BayesianNetwork(flip.getConnectedGraph()));
        optimizer.onGradientCalculation((point, gradient) -> {
            System.out.println("Gradient: @" + Arrays.toString(point) + " -> " + Arrays.toString(gradient));
        });

        optimizer.onFitnessCalculation((point, fitness) -> {
            System.out.println("Fitness @" + Arrays.toString(point) + " -> " + fitness);
        });

        optimizer.maxLikelihood();

        double[] actual = A.getValue().asFlatDoubleArray();
        assertArrayEquals(new double[]{8.0, -4.0}, actual, 0.1);
    }

}
