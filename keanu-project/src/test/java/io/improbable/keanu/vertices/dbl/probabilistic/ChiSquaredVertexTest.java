package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;

public class ChiSquaredVertexTest {

    private final Logger log = LoggerFactory.getLogger(ChiSquaredVertexTest.class);

    private Random random;

    @Before
    public void setup() {
        random = new Random(1);
    }

    @Test
    public void samplingProducesRealisticMeanAndStandardDeviation() {
        int N = 100000;
        double epsilon = 0.1;
        int k = 10;
        ChiSquaredVertex testChiVertex = new ChiSquaredVertex(new ConstantIntegerVertex(k), new Random(1));

        List<Double> samples = new ArrayList<>();
        for (int i = 0; i < N; i++) {
            double sample = testChiVertex.sample();
            samples.add(sample);
        }

        SummaryStatistics stats = new SummaryStatistics();
        samples.forEach(stats::addValue);

        double mean = stats.getMean();
        double sd = stats.getStandardDeviation();
        double standardDeviation = Math.sqrt(k * 2);
        log.info("Mean: " + mean);
        log.info("Standard deviation: " + sd);
        assertEquals(mean, k, epsilon);
        assertEquals(sd, standardDeviation, epsilon);
    }

    @Test
    public void chiSampleMethodMatchesDensityMethod() {
        Vertex<Double> vertex = new ChiSquaredVertex(
                new ConstantIntegerVertex(2),
                random
        );

        double from = 2;
        double to = 4;
        double bucketSize = 0.05;
        long sampleCount = 100000;

        ProbabilisticDoubleContract.sampleMethodMatchesDensityMethod(vertex, sampleCount, from, to, bucketSize, 1e-2);
    }

    @Test
    public void testLogDensityEqualsLogOfDensity() {
        ChiSquaredVertex chi = new ChiSquaredVertex(1);
        chi.setValue(0.0);
        double density = chi.density(0.1);
        double logDensity = chi.logDensity(0.1);

        Assert.assertEquals(Math.log(density), logDensity, 0.001);
    }

}
