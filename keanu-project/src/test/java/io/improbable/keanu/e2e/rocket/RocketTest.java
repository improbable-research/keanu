package io.improbable.keanu.e2e.rocket;

import io.improbable.keanu.algorithms.NetworkSample;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.algorithms.variational.optimizer.KeanuProbabilisticGraph;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.testcategory.Slow;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.generic.nonprobabilistic.ConditionalProbabilityTable;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import org.junit.experimental.categories.Category;

import java.util.Arrays;
import java.util.stream.Stream;

import static org.junit.Assert.assertEquals;

@Slf4j
public class RocketTest {

    @Category(Slow.class)
    @Test
    public void shouldYouLaunch() {

        KeanuRandom random = new KeanuRandom(1);

        BernoulliVertex oRingFailure = new BernoulliVertex(0.001);
        BernoulliVertex residualFuel = new BernoulliVertex(0.002);

        DoubleVertex overHeatProbability = ConditionalProbabilityTable.of(oRingFailure, residualFuel)
            .when(true, true).then(0.95)
            .when(false, true).then(0.29)
            .when(true, false).then(0.94)
            .orDefault(0.001);

        BooleanVertex overHeated = new BernoulliVertex(overHeatProbability);

        BernoulliVertex alarm1NotFalseNegative = new BernoulliVertex(0.99);
        BernoulliVertex alarm1FalsePositive = new BernoulliVertex(0.3);
        BooleanVertex alarm1 = overHeated.and(alarm1NotFalseNegative).or(alarm1FalsePositive);

        BernoulliVertex alarm2NotFalseNegative = new BernoulliVertex(0.95);
        BernoulliVertex alarm2FalsePositive = new BernoulliVertex(0.1);
        BooleanVertex alarm2 = overHeated.and(alarm2NotFalseNegative).or(alarm2FalsePositive);

        alarm1.observe(true);
        alarm2.observe(false);

        BayesianNetwork net = new BayesianNetwork(oRingFailure.getConnectedGraph());
        net.probeForNonZeroProbability(1000);

        Stream<NetworkSample> networkSamples = MetropolisHastings.withDefaultConfig().generatePosteriorSamples(
            new KeanuProbabilisticGraph(net),
            Arrays.asList(oRingFailure, residualFuel, alarm1FalsePositive)
        ).stream();

        long sampleCount = 10000;
        EventCounts eventCounts = networkSamples
            .limit(sampleCount)
            .map(state -> new EventCounts(
                state.get(oRingFailure).scalar(),
                state.get(residualFuel).scalar(),
                state.get(alarm1FalsePositive).scalar()
            )).reduce(new EventCounts(), EventCounts::combine);

        double posteriorProbOfORingFailure = eventCounts.oRingFailureCount / (double) sampleCount;
        double posteriorProbOfResidualFuel = eventCounts.residualFuelCount / (double) sampleCount;
        double posteriorProbOfAlarm1FalsePositive = eventCounts.alarm1FalsePositiveCount / (double) sampleCount;

        log.info("Probability that there is an ORing failure given the evidence: " + posteriorProbOfORingFailure);
        log.info("Probability that there is residual fuel given the evidence: " + posteriorProbOfResidualFuel);
        log.info("Probability that alarm1 is a false positive given the evidence: " + posteriorProbOfAlarm1FalsePositive);

        assertEquals(posteriorProbOfAlarm1FalsePositive, 1, 0.001);
    }

    private static class EventCounts {
        public long oRingFailureCount;
        public long residualFuelCount;
        public long alarm1FalsePositiveCount;

        public EventCounts() {
        }

        public EventCounts(boolean oRingFailure, boolean residualFuel, boolean alarm1FalsePositive) {
            oRingFailureCount = oRingFailure ? 1 : 0;
            residualFuelCount = residualFuel ? 1 : 0;
            alarm1FalsePositiveCount = alarm1FalsePositive ? 1 : 0;
        }

        public EventCounts combine(EventCounts that) {
            this.oRingFailureCount += that.oRingFailureCount;
            this.residualFuelCount += that.residualFuelCount;
            this.alarm1FalsePositiveCount += that.alarm1FalsePositiveCount;
            return this;
        }
    }

}
