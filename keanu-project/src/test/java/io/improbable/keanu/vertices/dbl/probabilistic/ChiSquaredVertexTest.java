package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.testcategory.Slow;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.LogProbGraph;
import io.improbable.keanu.vertices.LogProbGraphContract;
import io.improbable.keanu.vertices.LogProbGraphValueFeeder;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.distribution.ChiSquaredDistribution;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.categories.Category;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;

@Slf4j
public class ChiSquaredVertexTest {

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void logProbGraphMatchesKnownLogDensityOfScalar() {
        IntegerVertex k = ConstantVertex.of(1);
        ChiSquaredVertex vertex = new ChiSquaredVertex(k);
        LogProbGraph logProbGraph = vertex.logProbGraph();

        LogProbGraphValueFeeder.feedValue(logProbGraph, k, k.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, vertex, DoubleTensor.scalar(0.5));

        ChiSquaredDistribution chiSquaredDistribution = new ChiSquaredDistribution(1);
        double expectedLogDensity = chiSquaredDistribution.logDensity(0.5);

        LogProbGraphContract.matchesKnownLogDensity(logProbGraph, expectedLogDensity);
    }

    @Test
    public void logProbGraphMatchesKnownLogDensityOfVector() {
        IntegerVertex k = ConstantVertex.of(1, 1);
        ChiSquaredVertex vertex = new ChiSquaredVertex(k);
        LogProbGraph logProbGraph = vertex.logProbGraph();

        LogProbGraphValueFeeder.feedValue(logProbGraph, k, k.getValue());
        LogProbGraphValueFeeder.feedValue(logProbGraph, vertex, DoubleTensor.create(0.25, 0.75));

        ChiSquaredDistribution chiSquaredDistribution = new ChiSquaredDistribution(1);
        double expectedLogDensity = chiSquaredDistribution.logDensity(0.25) + chiSquaredDistribution.logDensity(0.75);

        LogProbGraphContract.matchesKnownLogDensity(logProbGraph, expectedLogDensity);
    }

    @Test
    public void samplingProducesRealisticMeanAndStandardDeviation() {
        int N = 100000;
        double epsilon = 0.1;
        int k = 10;
        ChiSquaredVertex testChiVertex = new ChiSquaredVertex(new long[]{N, 1}, k);

        SummaryStatistics stats = new SummaryStatistics();
        Arrays.stream(testChiVertex.sample(random).asFlatArray())
            .forEach(stats::addValue);

        double mean = stats.getMean();
        double sd = stats.getStandardDeviation();
        double standardDeviation = Math.sqrt(k * 2);
        log.info("Mean: " + mean);
        log.info("Standard deviation: " + sd);
        assertEquals(mean, k, epsilon);
        assertEquals(sd, standardDeviation, epsilon);
    }

    @Category(Slow.class)
    @Test
    public void chiSampleMethodMatchesLogProbMethod() {
        int sampleCount = 1000000;
        ChiSquaredVertex vertex = new ChiSquaredVertex(new long[]{sampleCount, 1}, 2);

        double from = 2;
        double to = 4;
        double bucketSize = 0.05;

        ProbabilisticDoubleTensorContract.sampleMethodMatchesLogProbMethod(
            vertex,
            from,
            to,
            bucketSize,
            1e-2,
            random
        );
    }

}
