package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import com.google.common.collect.ImmutableMap;
import io.improbable.keanu.algorithms.variational.optimizer.*;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.testcategory.Slow;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Test;
import org.junit.experimental.categories.Category;

import java.util.Map;
import java.util.stream.Collectors;

import static org.junit.Assert.assertEquals;

public class FitnessFunctionGradientTest {

    private final double dx = 0.0000000001;

    @Category(Slow.class)
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

        KeanuProbabilisticWithGradientGraph graph = new KeanuProbabilisticWithGradientGraph(new BayesianNetwork(A.getConnectedGraph()));

        FitnessFunction fitness = new LogProbFitnessFunction(graph);
        FitnessFunctionGradient fitnessGradient = new FitnessFunctionGradient(graph, false, (a, b) -> {
        });

        assert2DGradientEqualsApproxGradient(
            A.getReference(), B.getReference(),
            ImmutableMap.of(
                A.getReference(), DoubleTensor.scalar(5),
                B.getReference(), DoubleTensor.scalar(5)
            ),
            ImmutableMap.of(
                A.getReference(), DoubleTensor.scalar(0),
                B.getReference(), DoubleTensor.scalar(0)
            ),
            0.2,
            fitness,
            fitnessGradient
        );
    }

    @Category(Slow.class)
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

        KeanuProbabilisticWithGradientGraph graph = new KeanuProbabilisticWithGradientGraph(new BayesianNetwork(A.getConnectedGraph()));

        FitnessFunction fitness = new LogProbFitnessFunction(graph);
        FitnessFunctionGradient fitnessGradient = new FitnessFunctionGradient(graph, false, (a, b) -> {
        });

        assert2DGradientEqualsApproxGradient(
            A.getReference(), B.getReference(),
            ImmutableMap.of(
                A.getReference(), DoubleTensor.scalar(10),
                B.getReference(), DoubleTensor.scalar(10)
            ),
            ImmutableMap.of(
                A.getReference(), DoubleTensor.scalar(2),
                B.getReference(), DoubleTensor.scalar(2)
            ),
            0.2,
            fitness,
            fitnessGradient
        );
    }

    @Category(Slow.class)
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

        KeanuProbabilisticWithGradientGraph graph = new KeanuProbabilisticWithGradientGraph(new BayesianNetwork(A.getConnectedGraph()));
        FitnessFunction fitness = new LogProbFitnessFunction(graph);
        FitnessFunctionGradient fitnessGradient = new FitnessFunctionGradient(graph, false, (a, b) -> {
        });

        assert2DGradientEqualsApproxGradient(
            A.getReference(), B.getReference(),
            ImmutableMap.of(
                A.getReference(), DoubleTensor.scalar(5),
                B.getReference(), DoubleTensor.scalar(5)
            ),
            ImmutableMap.of(
                A.getReference(), DoubleTensor.scalar(0.1),
                B.getReference(), DoubleTensor.scalar(0.1)
            ),
            0.2,
            fitness,
            fitnessGradient
        );
    }


    /**
     * @param topRight   max input 1 and max input 2
     * @param bottomLeft min input 1 and min input 2
     * @param stepSize   step size for moving from max to min in both dimensions
     */
    private void assert2DGradientEqualsApproxGradient(VariableReference xRef,
                                                      VariableReference yRef,
                                                      Map<VariableReference, DoubleTensor> topRight,
                                                      Map<VariableReference, DoubleTensor> bottomLeft,
                                                      double stepSize,
                                                      FitnessFunction fitness,
                                                      FitnessFunctionGradient fitnessFunctionGradient) {

        Map<VariableReference, DoubleTensor> point = copyPoint(bottomLeft);

        int xStepCount = (int) ((topRight.get(xRef).minus(bottomLeft.get(xRef)).scalar()) / stepSize);
        int yStepCount = (int) ((topRight.get(yRef).minus(bottomLeft.get(yRef)).scalar()) / stepSize);

        for (int x = 0; x < xStepCount; x++) {

            for (int y = 0; y < yStepCount; y++) {

                Map<? extends VariableReference, DoubleTensor> gradient0 = fitnessFunctionGradient.value(point);

                double fitness0 = fitness.value(point);

                double da = dx;
                Map<VariableReference, DoubleTensor> pointA = ImmutableMap.of(
                    xRef, point.get(xRef).plus(da),
                    yRef, point.get(yRef)
                );

                double fitness1a = fitness.value(pointA);

                double approxGradientA = (fitness1a - fitness0) / da;
                double epsA = Math.max(Math.abs(gradient0.get(xRef).scalar() * 0.01), 0.0001);
                assertEquals(approxGradientA, gradient0.get(xRef).scalar(), epsA);

                double db = dx;

                Map<VariableReference, DoubleTensor> pointB = ImmutableMap.of(
                    xRef, point.get(xRef),
                    yRef, point.get(yRef).plus(db)
                );

                double fitness1b = fitness.value(pointB);

                double approxGradientB = (fitness1b - fitness0) / db;
                double epsB = Math.max(Math.abs(gradient0.get(yRef).scalar() * 0.01), 0.0001);
                assertEquals(approxGradientB, gradient0.get(yRef).scalar(), epsB);

                point.put(yRef, point.get(yRef).plus(stepSize));
            }

            point.put(xRef, point.get(xRef).plus(stepSize));
            point.put(yRef, bottomLeft.get(yRef));
        }

    }

    private Map<VariableReference, DoubleTensor> copyPoint(Map<VariableReference, DoubleTensor> point) {
        return point.entrySet().stream()
            .collect(Collectors.toMap(Map.Entry::getKey, e -> e.getValue().duplicate()));
    }

}
