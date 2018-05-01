package io.improbable.keanu.e2e.rocket;

import io.improbable.keanu.algorithms.sampling.RejectionSampler;
import io.improbable.keanu.network.BayesNet;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.probabilistic.Flip;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;

import static io.improbable.keanu.vertices.bool.BoolVertexTest.priorProbabilityTrue;
import static org.junit.Assert.assertEquals;

public class RocketTest {
    private final Logger log = LoggerFactory.getLogger(RocketTest.class);

    @Test
    public void shouldYouLaunch() {

        Random r = new Random(1);
        Flip oRingFailure = new Flip(0.001, r);

        Vertex<Boolean> oRingFailureCausesOverheat = new Flip(0.94, r);
        Vertex<Boolean> overHeatDueToORing = oRingFailure.and(oRingFailureCausesOverheat);

        Flip residualFuel = new Flip(0.002, r);

        Vertex<Boolean> residualFuelCausesOverheat = new Flip(0.29, r);
        Vertex<Boolean> overHeatDueToResidualFuel = residualFuel.and(residualFuelCausesOverheat);

        Vertex<Boolean> bothCauseOverheat = new Flip(0.95, r);
        Vertex<Boolean> overHeatDueToBoth = residualFuel
                .and(oRingFailure)
                .and(bothCauseOverheat);

        Flip overHeatDueToOther = new Flip(0.001, r);

        BoolVertex overHeated = overHeatDueToOther
                .or(overHeatDueToORing)
                .or(overHeatDueToResidualFuel)
                .or(overHeatDueToBoth);

        double probOfOverheat = priorProbabilityTrue(overHeated, 10000);
        log.info("Prior Probability rocket overheats: " + probOfOverheat);

        Flip alarm1NotFalseNegative = new Flip(0.99, r);
        Flip alarm1FalsePositive = new Flip(0.3, r);
        BoolVertex alarm1 = overHeated.and(alarm1NotFalseNegative).or(alarm1FalsePositive);

        Flip alarm2NotFalseNegative = new Flip(0.95, r);
        Flip alarm2FalsePositive = new Flip(0.1, r);
        BoolVertex alarm2 = overHeated.and(alarm2NotFalseNegative).or(alarm2FalsePositive);

        double probOfAlarm1 = priorProbabilityTrue(alarm1, 10000);
        log.info("Prior Probability alarm1 sounds: " + probOfAlarm1);

        double probOfAlarm2 = priorProbabilityTrue(alarm2, 10000);
        log.info("Prior Probability alarm2 sounds: " + probOfAlarm2);

        alarm1.observe(true);
        alarm2.observe(false);

        BayesNet net = new BayesNet(oRingFailure.getConnectedGraph());

        double posteriorProbOfORingFailure = RejectionSampler.getPosteriorProbability(net.getLatentVertices(), net.getObservedVertices(), () -> oRingFailure.getValue() == true, 10000);
        double posteriorProbOfResidualFuel = RejectionSampler.getPosteriorProbability(net.getLatentVertices(), net.getObservedVertices(), () -> residualFuel.getValue() == true, 10000);
        double posteriorProbOfAlarm1FalsePositive = RejectionSampler.getPosteriorProbability(net.getLatentVertices(), net.getObservedVertices(), () -> alarm1FalsePositive.getValue() == true, 10000);

        log.info("Probability that there is an ORing failure given the evidence: " + posteriorProbOfORingFailure);
        log.info("Probability that there is residual fuel given the evidence: " + posteriorProbOfResidualFuel);
        log.info("Probability that alarm1 is a false positive given the evidence: " + posteriorProbOfAlarm1FalsePositive);

        assertEquals(posteriorProbOfAlarm1FalsePositive, 1, 0.001);
    }

}
