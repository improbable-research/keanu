package io.improbable.keanu.algorithms.variational;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class FitnessFunctionWithGradientTest {

    private final double dx = 0.0000000001;

    @Test
    public void calculatesGradientCorrectlyWithMultiplyAndMinus() {

        GaussianVertex A = new GaussianVertex(1.0, 1.0);
        GaussianVertex B = new GaussianVertex(2.0, 1.0);

        A.setAndCascade(Nd4jDoubleTensor.scalar(1.5));
        B.setAndCascade(Nd4jDoubleTensor.scalar(2.5));

        DoubleVertex C = A.multiply(B);
        DoubleVertex D = A.minus(B);

        GaussianVertex cObservation = new GaussianVertex(C, 1.0);
        cObservation.observe(Nd4jDoubleTensor.scalar(3.0));

        GaussianVertex dObservation = new GaussianVertex(D, 1.0);
        dObservation.observe(Nd4jDoubleTensor.scalar(3.0));

        assert2DGradientEqualsApproxGradient(
            new double[]{5, 5},
            new double[]{0, 0},
            0.1,
            Arrays.asList(A, B, cObservation, dObservation),
            Arrays.asList(A, B)
        );
    }

    @Test
    public void calculatesGradientCorrectlyWithAdditionAndDivision() {

        GaussianVertex A = new GaussianVertex(7.0, 3.0);
        GaussianVertex B = new GaussianVertex(3.0, 3.0);

        A.setAndCascade(Nd4jDoubleTensor.scalar(6.0));
        B.setAndCascade(Nd4jDoubleTensor.scalar(3.0));

        DoubleVertex C = A.divideBy(B);
        DoubleVertex D = A.multiply(B);

        GaussianVertex cObservation = new GaussianVertex(C, 5.0);
        cObservation.observe(Nd4jDoubleTensor.scalar(2.1));

        GaussianVertex dObservation = new GaussianVertex(D, 5.0);
        dObservation.observe(Nd4jDoubleTensor.scalar(18.0));

        assert2DGradientEqualsApproxGradient(
            new double[]{10, 10},
            new double[]{2, 2},
            0.1,
            Arrays.asList(A, B, cObservation, dObservation),
            Arrays.asList(A, B)
        );
    }

    @Test
    public void calculatesGradientCorrectlyWithAdditionMultiplicationSubtractionDivision() {

        GaussianVertex A = new GaussianVertex(2.0, 3.0);
        GaussianVertex B = new GaussianVertex(3.0, 3.0);

        A.setAndCascade(Nd4jDoubleTensor.scalar(2.2));
        B.setAndCascade(Nd4jDoubleTensor.scalar(3.2));

        DoubleVertex C = A.plus(B);
        DoubleVertex D = A.multiply(B);

        DoubleVertex E = C.minus(D);
        DoubleVertex F = C.divideBy(D);

        GaussianVertex eObservation = new GaussianVertex(E, 5.0);
        eObservation.observe(Nd4jDoubleTensor.scalar(1.2));

        GaussianVertex fObservation = new GaussianVertex(F, C);
        fObservation.observe(Nd4jDoubleTensor.scalar(1.0));

        assert2DGradientEqualsApproxGradient(
            new double[]{5, 5},
            new double[]{0.1, 0.1},
            0.1,
            Arrays.asList(A, B, eObservation, fObservation),
            Arrays.asList(A, B)
        );
    }


    /**
     * @param topRight   max input 1 and max input 2
     * @param bottomLeft min input 1 and min input 2
     * @param stepSize   step size for moving from max to min in both dimensions
     */
    private void assert2DGradientEqualsApproxGradient(double[] topRight,
                                                      double[] bottomLeft,
                                                      double stepSize,
                                                      List<Vertex> probabilisticVertices,
                                                      List<? extends Vertex<DoubleTensor>> latentVertices) {

        FitnessFunctionWithGradient fitness = new FitnessFunctionWithGradient(probabilisticVertices, latentVertices);

        double[] point = Arrays.copyOf(bottomLeft, bottomLeft.length);

        int xStepCount = (int) ((topRight[0] - bottomLeft[0]) / stepSize);
        int yStepCount = (int) ((topRight[1] - bottomLeft[1]) / stepSize);

        for (int x = 0; x < xStepCount; x++) {

            for (int y = 0; y < yStepCount; y++) {

                double[] gradient0 = fitness.gradient().value(point);
                double fitness0 = fitness.fitness().value(point);

                double da = dx;
                double[] pointA = {point[0] + da, point[1]};

                double fitness1a = fitness.fitness().value(pointA);

                double approxGradientA = (fitness1a - fitness0) / da;
                double epsA = Math.max(Math.abs(gradient0[0] * 0.01), 0.0001);
                assertEquals(approxGradientA, gradient0[0], epsA);

                double db = dx;
                double[] pointB = {point[0], point[1] + db};

                double fitness1b = fitness.fitness().value(pointB);

                double approxGradientB = (fitness1b - fitness0) / db;
                double epsB = Math.max(Math.abs(gradient0[1] * 0.01), 0.0001);
                assertEquals(approxGradientB, gradient0[1], epsB);

                point[1] += stepSize;
            }

            point[0] += stepSize;
            point[1] = bottomLeft[1];
        }

    }

}
