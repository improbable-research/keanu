package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDoubleContract.moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues;
import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertEquals;

public class LogisticVertexTest {
    private final Logger log = LoggerFactory.getLogger(LogisticVertexTest.class);

    private static final double DELTA = 0.0001;

    @Test
    public void samplingProducesRealisticMeanAndStandardDeviation() {
        int N = (int) 1e6;
        double epsilon = 1e-2;
        double a = 0.0;
        double b = 1.0;
        double expectedVariance = (Math.pow(Math.PI, 2) / 3) * Math.pow(b, 2);
        LogisticVertex l = new LogisticVertex(a, b, new Random(1));

        List<Double> samples = new ArrayList<>();
        for (int i = 0; i < N; i++) {
            double sample = l.sample();
            samples.add(sample);
        }

        SummaryStatistics stats = new SummaryStatistics();
        samples.forEach(stats::addValue);

        double mean = stats.getMean();
        double sd = stats.getStandardDeviation();
        log.info("Mean: " + mean);
        log.info("Standard deviation: " + sd);
        assertEquals(mean, a, epsilon);
        assertEquals(sd, Math.sqrt(expectedVariance), epsilon);
    }

    @Test
    public void logDensityIsSameAsLogOfDensity() {
        LogisticVertex l = new LogisticVertex(new ConstantDoubleVertex(0.0), new ConstantDoubleVertex(2.0));
        double atValue = 0.5;
        double logOfDensity = Math.log(l.density(atValue));
        double logDensity = l.logDensity(atValue);
        assertEquals(logOfDensity, logDensity, 0.01);
    }

    @Test
    public void diffLnDensityIsSameAsLogOfDiffDensity() {
        LogisticVertex l = new LogisticVertex(new ConstantDoubleVertex(0.1), new ConstantDoubleVertex(1.0));
        double atValue = 0.5;
        l.setAndCascade(atValue);

        Map<String, Double> dP = l.dDensityAtValue();
        Map<String, Double> dlnP = l.dlnDensityAtValue();

        final double density = l.densityAtValue();
        for (String vertexId : dP.keySet()) {
            dP.put(vertexId, dP.get(vertexId) / density);
        }
        assertEquals(dP.get(l.getId()), dlnP.get(l.getId()), 0.01);
    }

    @Test
    public void gradientAtAIsZero() {
        double a = 0.0;
        double b = 0.5;
        LogisticVertex l = new LogisticVertex(a, b, new Random(1));
        l.setValue(a);
        double gradient = l.dDensityAtValue().get(l.getId());
        log.info("Gradient at a: " + gradient);
        assertEquals(gradient, 0, 0);
    }

    @Test
    public void gradientBeforeAIsPositive() {
        double a = 0.0;
        double b = 0.5;
        LogisticVertex l = new LogisticVertex(a, b, new Random(1));
        l.setValue(a - 1.0);
        double gradient = l.dDensityAtValue().get(l.getId());
        log.info("Gradient at x < a: " + gradient);
        assertTrue(gradient > 0);
    }

    @Test
    public void gradientAfterAIsNegative() {
        double a = 0.0;
        double b = 0.5;
        LogisticVertex l = new LogisticVertex(a, b, new Random(1));
        l.setValue(a + 1.0);
        double gradient = l.dDensityAtValue().get(l.getId());
        log.info("Gradient at x > a: " + gradient);
        assertTrue(gradient < 0);
    }

    @Test
    public void dDensityMatchesFiniteDifferenceCalculationFordPda() {
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
                DELTA);
    }

    @Test
    public void dDensityMatchesFiniteDifferenceCalculationFordPdb() {
        UniformVertex uniformB = new UniformVertex(new ConstantDoubleVertex(0.0), new ConstantDoubleVertex(1.));
        LogisticVertex l = new LogisticVertex(new ConstantDoubleVertex(0.0), uniformB);

        double vertexStartValue = 0.5;
        double vertexEndValue = 1.0;
        double vertexIncrement = 0.1;

        moveAlongDistributionAndTestGradientOnARangeOfHyperParameterValues(0.0,
                3.0,
                0.1,
                uniformB,
                l,
                vertexStartValue,
                vertexEndValue,
                vertexIncrement,
                DELTA);
    }

}
