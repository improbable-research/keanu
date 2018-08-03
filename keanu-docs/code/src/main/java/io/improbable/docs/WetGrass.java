package io.improbable.docs;

import java.util.Arrays;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.probabilistic.Flip;
import io.improbable.keanu.vertices.generic.nonprobabilistic.ConditionalProbabilityTable;
import io.improbable.keanu.vertices.generic.nonprobabilistic.If;

public class WetGrass {

    public static void main(String[] args) {

        //There's a simple 20% chance of rain and for the purposes
        //of this example, that doesn't depend on any other variables.
        BoolVertex rain = new Flip(0.2);

        //The probability of the sprinkler being on is dependent on
        //whether or not it has rained. It's very unlikely that the
        //sprinkler comes on if it's raining.
        BoolVertex sprinkler = new Flip(
            If.isTrue(rain)
                .then(0.01)
                .orElse(0.4)
        );

        //The grass being wet is dependent on whether or not it rained or
        //the sprinkler was on.
        BoolVertex wetGrass = new Flip(
            ConditionalProbabilityTable.of(sprinkler, rain)
                .when(false, false).then(1e-2)
                .when(false, true).then(0.8)
                .when(true, false).then(0.9)
                .orDefault(0.99)
        );

        //We observe that the grass is wet
        wetGrass.observe(true);

        //What does that observation say about the probability that it rained or that
        //the sprinkler was on?
        NetworkSamples posteriorSamples = MetropolisHastings.withDefaultConfig().getPosteriorSamples(
            new BayesianNetwork(wetGrass.getConnectedGraph()),
            Arrays.asList(sprinkler, rain),
            100000
        ).drop(10000).downSample(2);

        double probabilityOfRainGivenWetGrass = posteriorSamples.get(rain).probability(isRaining -> isRaining.scalar() == true);

        System.out.println(probabilityOfRainGivenWetGrass);
    }
}
