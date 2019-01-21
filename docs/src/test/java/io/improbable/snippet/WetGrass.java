package io.improbable.snippet;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.mcmc.KeanuMetropolisHastings;
import io.improbable.keanu.algorithms.variational.optimizer.KeanuProbabilisticModel;
import io.improbable.keanu.algorithms.variational.optimizer.ProbabilisticModel;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.generic.nonprobabilistic.ConditionalProbabilityTable;
import io.improbable.keanu.vertices.generic.nonprobabilistic.If;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.Set;

import static org.junit.Assert.assertEquals;

public class WetGrass {

    @Before
    public void deterministic() {
        KeanuRandom.setDefaultRandomSeed(1);
    }

    @Test
    public void doesFindCorrectProbability() {

        //%%SNIPPET_START%% Wetgrass
        //There's a simple 20% chance of rain and for the purposes
        //of this example, that doesn't depend on any other variables.
        BooleanVertex rain = new BernoulliVertex(0.2);

        //The probability of the sprinkler being on is dependent on
        //whether or not it has rained. It's very unlikely that the
        //sprinkler comes on if it's raining.
        BooleanVertex sprinkler = new BernoulliVertex(
            If.isTrue(rain)
                .then(0.01)
                .orElse(0.4)
        );

        //The grass being wet is dependent on whether or not it rained or
        //the sprinkler was on.
        // The following probabilities are the same as those used in Wikipedia article linked above.
        BooleanVertex wetGrass = new BernoulliVertex(
            ConditionalProbabilityTable.of(sprinkler, rain)
                .when(false, false).then(0.001)
                .when(false, true).then(0.8)
                .when(true, false).then(0.9)
                .orDefault(0.99)
        );

        //We observe that the grass is wet
        wetGrass.observe(true);

        //What does that observation say about the probability that it rained or that
        //the sprinkler was on?
        KeanuProbabilisticModel model = new KeanuProbabilisticModel(wetGrass.getConnectedGraph());
        NetworkSamples posteriorSamples = KeanuMetropolisHastings.withDefaultConfigFor(model).getPosteriorSamples(
            model,
            Arrays.asList(sprinkler, rain),
            100000
        ).drop(10000).downSample(2);

        double probabilityOfRainGivenWetGrass = posteriorSamples.get(rain).probability(isRaining -> isRaining.scalar() == true);

        System.out.println(probabilityOfRainGivenWetGrass);
        //%%SNIPPET_END%% Wetgrass

        assertEquals(0.358, probabilityOfRainGivenWetGrass, 0.1);
    }
}
