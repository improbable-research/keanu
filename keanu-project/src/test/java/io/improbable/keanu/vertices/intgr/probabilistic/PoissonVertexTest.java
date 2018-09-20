package io.improbable.keanu.vertices.intgr.probabilistic;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.closeTo;
import static org.junit.Assert.assertEquals;

import org.apache.commons.math3.distribution.PoissonDistribution;
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

        double logProb19 = poissonVertex.logPmf(19);
        double logProb20 = poissonVertex.logPmf(20);
        double logProb100 = poissonVertex.logPmf(100);

        PoissonDistribution distribution = new PoissonDistribution(25);
        assertThat(logProb19, closeTo(distribution.logProbability(19), 1e-6));
        assertThat(logProb20, closeTo(distribution.logProbability(20), 1e-6));
        assertThat(logProb100, closeTo(distribution.logProbability(100), 1e-6));
    }
}
