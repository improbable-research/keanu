package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.algorithms.variational.GradientOptimizer;
import io.improbable.keanu.network.BayesNet;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import org.junit.Before;
import org.junit.Test;

import static junit.framework.TestCase.assertEquals;

public class SmoothUniformTest {

    DoubleVertex A;
    DoubleVertex B;
    DoubleVertex C;
    DoubleVertex CObserved;
    KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void optimizerMovesAwayFromLeftShoulder() {

        A = new SmoothUniformVertex(0.0, 1000., 0.05);
        B = new SmoothUniformVertex(0, 1000);
        C = A.plus(B);
        CObserved = new GaussianVertex(C, 0.2);

        //Both on left shoulder
        A.setAndCascade(-25.0);
        B.setAndCascade(-9.0);

        CObserved.observe(50.0);
        shouldFind(50.0);
    }

    @Test
    public void optimizerMovesAwayFromRightShoulder() {

        A = new SmoothUniformVertex(-1000, 0);
        B = new SmoothUniformVertex(-1000, 0);
        C = A.plus(B);
        CObserved = new GaussianVertex(C, 0.2);

        //Both on right shoulder
        A.setAndCascade(5.0);
        B.setAndCascade(9.0);

        double expected = -50.0;
        CObserved.observe(expected);
        shouldFind(expected);
    }

    private void shouldFind(double expected) {
        GradientOptimizer optimizer = new GradientOptimizer(new BayesNet(A.getConnectedGraph()));
        optimizer.maxAPosteriori(1000);
        assertEquals(expected, C.getValue(), 0.001);
    }

    @Test
    public void smoothUniformSampleMethodMatchesLogProbMethod() {

        double edgeSharpness = 1.0;
        SmoothUniformVertex uniform = new SmoothUniformVertex(
            0.0,
            1.0,
            edgeSharpness
        );

        double from = -1;
        double to = 2;
        double delta = 0.05;
        long N = 1000000;

        ProbabilisticDoubleContract.sampleMethodMatchesLogProbMethod(
            uniform,
            N,
            from,
            to,
            delta,
            1e-2,
            random
        );
    }

    @Test
    public void isTreatedAsConstantWhenObserved() {
        SmoothUniformVertex vertexUnderTest = new SmoothUniformVertex(
            new UniformVertex(0.0, 1.0),
            3.0,
            0.01
        );
        ProbabilisticDoubleContract.isTreatedAsConstantWhenObserved(vertexUnderTest);
        ProbabilisticDoubleContract.hasNoGradientWithRespectToItsValueWhenObserved(vertexUnderTest);
    }

}
