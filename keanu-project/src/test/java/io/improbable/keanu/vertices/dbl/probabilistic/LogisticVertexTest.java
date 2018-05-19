package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

import static io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDoubleContract.moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues;
import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertEquals;

public class LogisticVertexTest {

    private static final double DELTA = 0.0001;

    @Rule
    public DeterministicRule deterministicRule = new DeterministicRule();
    private final Logger log = LoggerFactory.getLogger(LogisticVertexTest.class);
    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void samplingProducesRealisticMeanAndStandardDeviation() {
        int N = (int) 1e6;
        double epsilon = 1e-2;
        double a = 0.0;
        double b = 1.0;

        LogisticVertex logisticVertex = new LogisticVertex(a, b);

        double mean = a;
        double standardDeviation = Math.sqrt((Math.pow(Math.PI, 2) / 3) * Math.pow(b, 2));

        ProbabilisticDoubleContract.samplingProducesRealisticMeanAndStandardDeviation(
            N,
            logisticVertex,
            mean,
            standardDeviation,
            epsilon,
            random
        );
    }

    @Test
    public void gradientAtAIsZero() {
        double a = 0.0;
        double b = 0.5;
        LogisticVertex logisticVertex = new LogisticVertex(a, b);
        logisticVertex.setValue(a);
        double gradient = logisticVertex.dLogProbAtValue().get(logisticVertex.getId()).scalar();
        log.info("Gradient at a: " + gradient);
        assertEquals(gradient, 0, 0);
    }

    @Test
    public void gradientBeforeAIsPositive() {
        double a = 0.0;
        double b = 0.5;
        LogisticVertex logisticVertex = new LogisticVertex(a, b);
        logisticVertex.setValue(a - 1.0);
        double gradient = logisticVertex.dLogProbAtValue().get(logisticVertex.getId()).scalar();
        log.info("Gradient at x < a: " + gradient);
        assertTrue(gradient > 0);
    }

    @Test
    public void gradientAfterAIsNegative() {
        double a = 0.0;
        double b = 0.5;
        LogisticVertex logisticVertex = new LogisticVertex(a, b);
        logisticVertex.setValue(a + 1.0);
        double gradient = logisticVertex.dLogProbAtValue().get(logisticVertex.getId()).scalar();
        log.info("Gradient at x > a: " + gradient);
        assertTrue(gradient < 0);
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPda() {
        UniformVertex uniformA = new UniformVertex(new ConstantDoubleVertex(0.), new ConstantDoubleVertex(1.));
        LogisticVertex l = new LogisticVertex(uniformA, new ConstantDoubleVertex(1.0));

        double vertexStartValue = 0.5;
        double vertexEndValue = 5.0;
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(0.0,
            0.9,
            0.1,
            uniformA,
            l,
            vertexStartValue,
            vertexEndValue,
            vertexIncrement,
            DELTA
        );
    }

    @Test
    public void isTreatedAsConstantWhenObserved() {
        LogisticVertex vertexUnderTest = new LogisticVertex(
            new UniformVertex(0.0, 1.0),
            3.0
        );
        ProbabilisticDoubleContract.isTreatedAsConstantWhenObserved(vertexUnderTest);
        ProbabilisticDoubleContract.hasNoGradientWithRespectToItsValueWhenObserved(vertexUnderTest);
    }

    @Test
    public void dLogProbMatchesFiniteDifferenceCalculationFordPdb() {
        UniformVertex uniformB = new UniformVertex(0.0, 1.);
        LogisticVertex logisticVertex = new LogisticVertex(0.0, uniformB);

        double vertexStartValue = 0.5;
        double vertexEndValue = 1.0;
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(
            0.5,
            3.0,
            0.1,
            uniformB,
            logisticVertex,
            vertexStartValue,
            vertexEndValue,
            vertexIncrement,
            DELTA);
    }

    @Test
    public void inferHyperParamsFromSamples() {

        double trueA = 2.0;
        double trueB = 2.0;

        List<DoubleVertex> AB = new ArrayList<>();
        AB.add(new ConstantDoubleVertex(trueA));
        AB.add(new ConstantDoubleVertex(trueB));

        List<DoubleVertex> latentAB = new ArrayList<>();
        latentAB.add(new SmoothUniformVertex(0.01, 10.0));
        latentAB.add(new SmoothUniformVertex(0.01, 10.0));

        VertexVariationalMAP.inferHyperParamsFromSamples(
            hyperParams -> new LogisticVertex(hyperParams.get(0), hyperParams.get(1)),
            AB,
            latentAB,
            1000,
            random
        );
    }

}
