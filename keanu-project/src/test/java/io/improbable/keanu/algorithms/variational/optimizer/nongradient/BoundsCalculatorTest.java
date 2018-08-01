package io.improbable.keanu.algorithms.variational.optimizer.nongradient;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple.ConcatenationVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.apache.commons.math3.optim.SimpleBounds;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;

public class BoundsCalculatorTest {

    @Test
    public void calculatesBoundsWhenAllAreSpecified() {

        DoubleVertex A = new UniformVertex(new int[]{1, 2}, -2, 1);
        DoubleVertex B = new UniformVertex(new int[]{1, 2}, -2, 1);
        B.observe(2);
        DoubleVertex D = A.plus(B);

        OptimizerBounds bounds = new OptimizerBounds();
        bounds.addBound(A, DoubleTensor.create(new double[]{-2, -1}), 1);

        ApacheMathSimpleBoundsCalculator boundsCalculator = new ApacheMathSimpleBoundsCalculator(Double.POSITIVE_INFINITY, bounds);

        ImmutableList<DoubleVertex> latentVertices = ImmutableList.of(A);
        SimpleBounds simpleBounds = boundsCalculator.getBounds(latentVertices, new double[]{0, 0, 0});

        assertArrayEquals(new double[]{-2, -1}, simpleBounds.getLower(), 0.0);
        assertArrayEquals(new double[]{1, 1}, simpleBounds.getUpper(), 0.0);
    }

    @Test
    public void calculatesBoundsWhenNotAllAreSpecified() {

        DoubleVertex A = new UniformVertex(new int[]{1, 2}, -2, 1);
        DoubleVertex B = new UniformVertex(new int[]{1, 2}, -2, 1);
        B.observe(new double[]{2, 3});
        DoubleVertex C = new GaussianVertex(new int[]{1, 2}, 0, 1);
        DoubleVertex D = A.plus(B);
        DoubleVertex E = C.plus(D);
        DoubleVertex F = new GaussianVertex(new int[]{2, 2}, new ConcatenationVertex(0, E, D), 1);

        OptimizerBounds bounds = new OptimizerBounds();
        bounds.addBound(A, DoubleTensor.create(new double[]{-2, -1}), 1);
        bounds.addBound(F,
            DoubleTensor.create(new double[]{-2, -1, -3, -4}, new int[]{2, 2}),
            DoubleTensor.create(new double[]{2, 1, 3, 4}, new int[]{2, 2})
        );

        ImmutableList<DoubleVertex> latentVertices = ImmutableList.of(A, C, F);

        ApacheMathSimpleBoundsCalculator boundsInfCalculator = new ApacheMathSimpleBoundsCalculator(Double.POSITIVE_INFINITY, bounds);
        SimpleBounds boundsA = boundsInfCalculator.getBounds(latentVertices, new double[]{0, 0, 0, 0, 0, 0, 0, 0});

        assertArrayEquals(
            new double[]{-2, -1, Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY, -2, -1, -3, -4},
            boundsA.getLower(),
            0.0
        );

        assertArrayEquals(
            new double[]{1, 1, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, 2, 1, 3, 4},
            boundsA.getUpper(),
            0.0
        );

        ApacheMathSimpleBoundsCalculator bounds20Calculator = new ApacheMathSimpleBoundsCalculator(20, bounds);
        SimpleBounds boundsB = bounds20Calculator.getBounds(latentVertices, new double[]{0, 0, 0, 0, 0, 0, 0, 0});

        assertArrayEquals(new double[]{-2, -1, -20, -20, -2, -1, -3, -4}, boundsB.getLower(), 0.0);
        assertArrayEquals(new double[]{1, 1, 20, 20, 2, 1, 3, 4}, boundsB.getUpper(), 0.0);
    }
}
