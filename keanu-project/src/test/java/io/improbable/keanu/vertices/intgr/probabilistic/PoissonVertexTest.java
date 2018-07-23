package io.improbable.keanu.vertices.intgr.probabilistic;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import io.improbable.keanu.distributions.dual.ParameterName;
import io.improbable.keanu.tensor.TensorShapeException;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.MissingParameterException;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.DistributionVertexBuilder;
import io.improbable.keanu.vertices.dbl.probabilistic.VertexOfType;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;

public class PoissonVertexTest {
    private final Logger log = LoggerFactory.getLogger(PoissonVertexTest.class);

    @Test
    public void samplingProducesRealisticMeanAndStandardDeviation() {
        int N = 100000;
        double epsilon = 0.1;
        Double mu = 10.0;
        KeanuRandom random = new KeanuRandom(1);
        PoissonVertex testPoissonVertex = VertexOfType.poisson(mu);

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

        PoissonVertex poissonVertex = new DistributionVertexBuilder()
        .withInput(ParameterName.MU, mu)
            .poisson();

        double logProb = poissonVertex.logProb(IntegerTensor.scalar(19));
        double logProbThreshold = poissonVertex.logProb(IntegerTensor.scalar(20));
        double logProbAboveThreshold = poissonVertex.logProb(IntegerTensor.scalar(21));

        assertTrue(logProbAboveThreshold > logProbThreshold && logProbThreshold > logProb);
    }


    @Test(expected = TensorShapeException.class)
    public void itThrowsIfTheInputDimensionsDontMatch() {
        new DistributionVertexBuilder()
            .withInput(ParameterName.P, new ConstantDoubleVertex(new double[] {1.,2.,3.}))
            .withInput(ParameterName.N, new ConstantIntegerVertex(new int[] {1,2}))
            .binomial();
    }

    @Test(expected = MissingParameterException.class)
    public void itThrowsIfYouHaventSetParameterN() {
        new DistributionVertexBuilder()
            .shaped(1,2,3)
            .withInput(ParameterName.P, new ConstantDoubleVertex(new double[] {1.,2.,3.}))
            .binomial();
    }

    @Test(expected = MissingParameterException.class)
    public void itThrowsIfYouHaventSetParameterP() {
        new DistributionVertexBuilder()
            .shaped(1,2,3)
            .withInput(ParameterName.N, new ConstantIntegerVertex(new int[] {1,2}))
            .binomial();
    }
}
