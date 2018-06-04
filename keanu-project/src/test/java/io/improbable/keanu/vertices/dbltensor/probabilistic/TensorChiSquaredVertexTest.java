package io.improbable.keanu.vertices.dbltensor.probabilistic;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class TensorChiSquaredVertexTest {

    private final Logger log = LoggerFactory.getLogger(TensorChiSquaredVertexTest.class);

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void samplingProducesRealisticMeanAndStandardDeviation() {
        int N = 100000;
        double epsilon = 0.1;
        int k = 10;
        TensorChiSquaredVertex testChiVertex = new TensorChiSquaredVertex(new int[]{N, 1}, k);

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

    @Test
    public void chiSampleMethodMatchesLogProbMethod() {
        int sampleCount = 1000000;
        Vertex<DoubleTensor> vertex = new TensorChiSquaredVertex(new int[]{sampleCount, 1}, 2);

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
