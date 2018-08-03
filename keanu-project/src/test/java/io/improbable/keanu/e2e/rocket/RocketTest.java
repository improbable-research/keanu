package io.improbable.keanu.e2e.rocket;

import io.improbable.keanu.algorithms.sampling.RejectionSampler;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.probabilistic.Flip;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static io.improbable.keanu.vertices.bool.BoolVertexTest.priorProbabilityTrue;
import static org.junit.Assert.assertEquals;

public class RocketTest {
    private final Logger log = LoggerFactory.getLogger(RocketTest.class);

    @Test
    public void shouldYouLaunch() {

        KeanuRandom random = new KeanuRandom(1);
        Flip oRingFailure = new Flip(0.001);

        Vertex<BooleanTensor> oRingFailureCausesOverheat = new Flip(0.94);
        Vertex<BooleanTensor> overHeatDueToORing = oRingFailure.and(oRingFailureCausesOverheat);

        Flip residualFuel = new Flip(0.002);

        Vertex<BooleanTensor> residualFuelCausesOverheat = new Flip(0.29);
        Vertex<BooleanTensor> overHeatDueToResidualFuel = residualFuel.and(residualFuelCausesOverheat);

        Vertex<BooleanTensor> bothCauseOverheat = new Flip(0.95);
        Vertex<BooleanTensor> overHeatDueToBoth = residualFuel
            .and(oRingFailure)
            .and(bothCauseOverheat);

        Flip overHeatDueToOther = new Flip(0.001);

        BoolVertex overHeated = overHeatDueToOther
            .or(overHeatDueToORing)
            .or(overHeatDueToResidualFuel)
            .or(overHeatDueToBoth);

        double probOfOverheat = priorProbabilityTrue(overHeated, 10000, random);
        log.info("Prior Probability rocket overheats: " + probOfOverheat);

        Flip alarm1NotFalseNegative = new Flip(0.99);
        Flip alarm1FalsePositive = new Flip(0.3);
        BoolVertex alarm1 = overHeated.and(alarm1NotFalseNegative).or(alarm1FalsePositive);

        Flip alarm2NotFalseNegative = new Flip(0.95);
        Flip alarm2FalsePositive = new Flip(0.1);
        BoolVertex alarm2 = overHeated.and(alarm2NotFalseNegative).or(alarm2FalsePositive);

        double probOfAlarm1 = priorProbabilityTrue(alarm1, 10000, random);
        log.info("Prior Probability alarm1 sounds: " + probOfAlarm1);

        double probOfAlarm2 = priorProbabilityTrue(alarm2, 10000, random);
        log.info("Prior Probability alarm2 sounds: " + probOfAlarm2);

        alarm1.observe(true);
        alarm2.observe(false);

        BayesianNetwork net = new BayesianNetwork(oRingFailure.getConnectedGraph());

        double posteriorProbOfORingFailure = RejectionSampler.getPosteriorProbability(
            net.getLatentVertices(),
            net.getObservedVertices(),
            () -> oRingFailure.getValue().scalar() == true,
            10000,
            random
        );
        double posteriorProbOfResidualFuel = RejectionSampler.getPosteriorProbability(
            net.getLatentVertices(),
            net.getObservedVertices(),
            () -> residualFuel.getValue().scalar() == true,
            10000,
            random
        );

        double posteriorProbOfAlarm1FalsePositive = RejectionSampler.getPosteriorProbability(
            net.getLatentVertices(),
            net.getObservedVertices(),
            () -> alarm1FalsePositive.getValue().scalar() == true,
            10000,
            random
        );

        log.info("Probability that there is an ORing failure given the evidence: " + posteriorProbOfORingFailure);
        log.info("Probability that there is residual fuel given the evidence: " + posteriorProbOfResidualFuel);
        log.info("Probability that alarm1 is a false positive given the evidence: " + posteriorProbOfAlarm1FalsePositive);

        assertEquals(posteriorProbOfAlarm1FalsePositive, 1, 0.001);
    }

}
