package io.improbable.keanu.vertices.intgr.probabilistic;

import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import org.apache.commons.math3.distribution.PoissonDistribution;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class PoissonVertexTest {
    private final Logger log = LoggerFactory.getLogger(PoissonVertexTest.class);

    @Test
    public void samplingProducesRealisticMeanAndStandardDeviation() {
        int N = 100000;
        double epsilon = 0.1;
        Double mu = 10.0;
        KeanuRandom random = new KeanuRandom(1);
        PoissonVertex testPoissonVertex = new PoissonVertex(mu);

        List<Integer> samples = new ArrayList<>();
        for (int i = 0; i < N; i++) {
            Integer sample = testPoissonVertex.sample(random).scalar();
            samples.add(sample);
        }

        SummaryStatistics stats = new SummaryStatistics();
        samples.forEach(stats::addValue);

        double mean = stats.getMean();
        double sd = stats.getStandardDeviation();
        double standardDeviation = Math.sqrt(mu);
        log.info("Mean: " + mean);
        log.info("Standard deviation: " + sd);
        assertEquals(mean, mu, epsilon);
        assertEquals(sd, standardDeviation, epsilon);
    }


    @Test
    public void logProbForValuesGreaterThanTwenty() {
        double mu = 25.0;

        PoissonVertex poissonVertex = new PoissonVertex(mu);

        double logProb = poissonVertex.logPmf(19);
        double logProbThreshold = poissonVertex.logPmf(20);
        double logProbAboveThreshold = poissonVertex.logPmf(21);

        assertTrue(logProbAboveThreshold > logProbThreshold && logProbThreshold > logProb);
    }

    @Test
    public void logProbMatchesKnownLogProb() {
        DoubleVertex mu = ConstantVertex.of(new double[]{5, 7});
        PoissonVertex poissonVertex = new PoissonVertex(mu);
        double actualLogProb = poissonVertex.logPmf(new int[]{39, 49});

        PoissonDistribution distribution1 = new PoissonDistribution(5);
        PoissonDistribution distribution2 = new PoissonDistribution(7);
        double expectedLogProb = distribution1.logProbability(39) + distribution2.logProbability(49);

        assertEquals(expectedLogProb, actualLogProb, 1e-5);
    }
}
