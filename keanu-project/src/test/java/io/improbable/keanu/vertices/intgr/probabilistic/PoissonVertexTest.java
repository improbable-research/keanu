package io.improbable.keanu.vertices.intgr.probabilistic;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class PoissonVertexTest {
    private final Logger log = LoggerFactory.getLogger(PoissonVertexTest.class);

    @Test
    public void samplingProducesRealisticMeanAndStandardDeviation() {
        int N = 100000;
        double epsilon = 0.1;
        Double mu = 25.0;
        KeanuRandom random = new KeanuRandom(1);
        PoissonVertex testPoissonVertex = new PoissonVertex(mu);

        SummaryStatistics stats = new SummaryStatistics();
        for (int i = 0; i < N; i++) {
            Integer sample = testPoissonVertex.sample(random).scalar();
            stats.addValue(sample);
        }

        double mean = stats.getMean();
        double sd = stats.getStandardDeviation();
        double standardDeviation = Math.sqrt(mu);
        log.info("Mean: " + mean);
        log.info("Standard deviation: " + sd);
        assertEquals(mean, mu, epsilon);
        assertEquals(sd, standardDeviation, epsilon);
    }

    /*
     * Certain implementations of Poisson sample generation are susceptible to numerical stability issues.  In certain
     * cases e ^ (- mu) is calculated and used as a stopping condition in a loop.  With large mu (~800) though the
     * calculated values bottom out at 0.0 leading to infinite loops.  This test just ensures that we don't hit this
     * issue.
     */
    @Test
    public void largeMuIsSupported() {
        final double mu = 900.0;
        PoissonVertex testVertex = new PoissonVertex(mu);

        /*
         * We don't actually need the output value here - we're simply checking that high mu sample function returns in
         * a timely fashion.
         */
        int sample = testVertex.sample().scalar();
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
}
