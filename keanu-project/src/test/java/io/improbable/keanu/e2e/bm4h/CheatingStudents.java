package io.improbable.keanu.e2e.bm4h;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.probabilistic.Flip;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.If;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.probabilistic.BinomialVertex;
import org.junit.Rule;
import org.junit.Test;

import java.util.Collections;

import static org.junit.Assert.assertEquals;

public class CheatingStudents {

    @Rule
    public DeterministicRule rule = new DeterministicRule();

    private final int numberOfStudents = 100;
    private final int numberOfYesAnswers = 35;

    @Test
    public void doesWorkWithHigherDimensionDescription() {

        int numberOfSamples = 10000;
        UniformVertex probabilityOfCheating = new UniformVertex(0.0, 1.0);
        BoolVertex studentCheated = new Flip(new int[]{1, numberOfStudents}, probabilityOfCheating);
        BoolVertex answerIsTrue = new Flip(new int[]{1, numberOfStudents}, 0.5);
        BoolVertex randomAnswer = new Flip(new int[]{1, numberOfStudents}, 0.5);

        DoubleVertex answer = If.isTrue(answerIsTrue)
            .then(
                If.isTrue(studentCheated)
                    .then(1)
                    .orElse(0)
            ).orElse(
                If.isTrue(randomAnswer)
                    .then(1)
                    .orElse(0)
            );

        DoubleVertex answerTotal = new GaussianVertex(answer.sum(), 1);
        answerTotal.observe(numberOfYesAnswers);

        BayesianNetwork network = new BayesianNetwork(answerTotal.getConnectedGraph());

        NetworkSamples networkSamples = MetropolisHastings.withDefaultConfig().getPosteriorSamples(
            network,
            Collections.singletonList(probabilityOfCheating),
            numberOfSamples
        ).drop(numberOfSamples / 10).downSample(network.getLatentVertices().size());

        double approximateProbabilityOfCheating = networkSamples
            .getDoubleTensorSamples(probabilityOfCheating)
            .getAverages()
            .scalar();

        assertEquals(0.2, approximateProbabilityOfCheating, 0.05);
    }

    @Test
    public void doesWorkWithBinomial() {
        int numberOfSamples = 10000;

        UniformVertex probabilityOfCheating = new UniformVertex(0.0, 1.0);
        DoubleVertex pYesAnswer = probabilityOfCheating.times(0.5).plus(0.25);
        IntegerVertex answerTotal = new BinomialVertex(pYesAnswer, numberOfStudents);
        answerTotal.observe(numberOfYesAnswers);

        BayesianNetwork network = new BayesianNetwork(answerTotal.getConnectedGraph());

        NetworkSamples networkSamples = MetropolisHastings.withDefaultConfig().getPosteriorSamples(
            network,
            Collections.singletonList(probabilityOfCheating),
            numberOfSamples
        ).drop(numberOfSamples / 10).downSample(network.getLatentVertices().size());

        double approximateProbabilityOfCheating = networkSamples
            .getDoubleTensorSamples(probabilityOfCheating)
            .getAverages()
            .scalar();

        assertEquals(0.2, approximateProbabilityOfCheating, 0.05);
    }

}
