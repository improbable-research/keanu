package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;

public class LaplaceVertexTest {

    private final Logger log = LoggerFactory.getLogger(LaplaceVertexTest.class);

    @Test
    public void laplaceSampleMethodMatchesDensityMethod() {
        int N = 100000;
        double epsilon = 0.01;
        LaplaceVertex l = new LaplaceVertex(new ConstantDoubleVertex(0.0), new ConstantDoubleVertex(1.0), new Random(1));

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
        assertEquals(0.0, mean, epsilon);
        assertEquals(Math.sqrt(2 * 1.0), sd, epsilon);
    }

    @Test
    public void samplingMatchesPdf() {
        LaplaceVertex laplace = new LaplaceVertex(
                new ConstantDoubleVertex(0.0),
                new ConstantDoubleVertex(1.0),
                new Random()
        );

        ProbabilisticDoubleContract.sampleMethodMatchesDensityMethod(
                laplace,
                100000,
                2.0,
                10.0,
                0.1,
                0.01);
    }

    @Test
    public void logDensityIsSameAsLogOfDensity() {
        LaplaceVertex l = new LaplaceVertex(new ConstantDoubleVertex(0.0), new ConstantDoubleVertex(1.0));
        double atValue = 0.5;
        double logOfDensity = Math.log(l.density(atValue));
        double logDensity = l.logDensity(atValue);
        assertEquals(logDensity, logOfDensity, 0.01);
    }
}
