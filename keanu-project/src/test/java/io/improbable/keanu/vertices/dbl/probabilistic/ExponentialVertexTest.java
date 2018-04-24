package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDoubleContract.moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues;
import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertEquals;

public class ExponentialVertexTest {

    private final Logger log = LoggerFactory.getLogger(ExponentialVertexTest.class);

    private static final double DELTA = 0.0001;

    private Random random;

    @Before
    public void setup() {
        random = new Random(1);
    }

    @Test
    public void samplingProducesRealisticMeanAndStandardDeviation() {
        int N = 100000;
        double epsilon = 1e-2;

        double a = 0.0;
        double b = 0.5;

        ExponentialVertex e = new ExponentialVertex(a, b, new Random(1));

        double mean = Math.pow(1 / b, -1);
        double standardDeviation = Math.sqrt(Math.pow(1 / b, -2));

        ProbabilisticDoubleContract.samplingProducesRealisticMeanAndStandardDeviation(N, e, mean, standardDeviation, epsilon);
    }

    @Test
    public void logDensityIsSameAsLogOfDensity() {
        ExponentialVertex e = new ExponentialVertex(new ConstantDoubleVertex(0.0), new ConstantDoubleVertex(2.0));
        double atValue = 0.5;
        double logOfDensity = Math.log(e.density(atValue));
        double logDensity = e.logDensity(atValue);
        assertEquals(logOfDensity, logDensity, 0.01);
    }

    @Test
    public void diffLnDensityIsSameAsLogOfDiffDensity() {
        ExponentialVertex e = new ExponentialVertex(new ConstantDoubleVertex(0.0), new ConstantDoubleVertex(1.0));
        ProbabilisticDoubleContract.diffLnDensityIsSameAsLogOfDiffDensity(e, 0.5, 0.001);
    }

    @Test
    public void gradientAtAIsMinusOne() {
        double a = 0.0;
        double b = 1.0;
        ExponentialVertex e = new ExponentialVertex(a, b, new Random(1));
        e.setValue(a);
        double gradient = e.dDensityAtValue().get(e.getId());
        log.info("Gradient at a: " + gradient);
        assertEquals(-1, gradient, 0);
    }

    @Test
    public void gradientContinuesToIncreaseAsValueIncreases() {
        ExponentialVertex exponentialVertex = new ExponentialVertex(0, 1, new Random(1));
        int n = 100;
        double value = 0.0;
        double step = 0.1;
        exponentialVertex.setValue(value);
        double initialGradient = exponentialVertex.dDensityAtValue().get(exponentialVertex.getId());

        for (int i = 0; i < n; i++) {
            exponentialVertex.setValue(value += step);
            double gradient = exponentialVertex.dDensityAtValue().get(exponentialVertex.getId());
            assertTrue(gradient > initialGradient);
            initialGradient = gradient;
        }
    }

    @Test
    public void dDensityMatchesFiniteDifferenceCalculationFordPda() {
        UniformVertex uniformA = new UniformVertex(new ConstantDoubleVertex(0.), new ConstantDoubleVertex(1.));
        ExponentialVertex exp = new ExponentialVertex(uniformA, new ConstantDoubleVertex(1.0));

        double vertexStartValue = 1.0;
        double vertexEndValue = 5.0;
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(0.0,
                0.5,
                0.1,
                uniformA,
                exp,
                vertexStartValue,
                vertexEndValue,
                vertexIncrement,
                DELTA);
    }

    @Test
    public void dDensityMatchesFiniteDifferenceCalculationFordPdb() {
        UniformVertex uniformB = new UniformVertex(new ConstantDoubleVertex(0.0), new ConstantDoubleVertex(1.));
        ExponentialVertex exp = new ExponentialVertex(new ConstantDoubleVertex(0.0), uniformB);

        double vertexStartValue = 0.5;
        double vertexEndValue = 1.0;
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(1.0,
                3.0,
                0.1,
                uniformB,
                exp,
                vertexStartValue,
                vertexEndValue,
                vertexIncrement,
                DELTA);
    }

    @Test
    public void inferHyperParamsFromSamples() {

        double trueB = 2.0;

        DoubleVertex A = new ConstantDoubleVertex(0.0);

        List<DoubleVertex> AB = new ArrayList<>();
        AB.add(A);
        AB.add(new ConstantDoubleVertex(trueB));

        List<DoubleVertex> latentAB = new ArrayList<>();
        latentAB.add(A);
        latentAB.add(new SmoothUniformVertex(0.01, 10.0, random));

        VertexVariationalMAPTest.inferHyperParamsFromSamples(
                hyperParams -> new ExponentialVertex(hyperParams.get(0), hyperParams.get(1), random),
                AB,
                latentAB,
                10000
        );
    }

}
