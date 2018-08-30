package io.improbable.docs;

import java.util.Arrays;

import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.algorithms.mcmc.NetworkSamplesGenerator;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.ConditionalProbabilityTable;
import io.improbable.keanu.vertices.generic.nonprobabilistic.If;

public class WetGrass {

    public static void main(String[] args) {

        //There's a simple 20% chance of rain and for the purposes
        //of this example, that doesn't depend on any other variables.
        BoolVertex rain = new BernoulliVertex(0.2);

        //The probability of the sprinkler being on is dependent on
        //whether or not it has rained. It's very unlikely that the
        //sprinkler comes on if it's raining.
        BoolVertex sprinkler = new BernoulliVertex(
            If.isTrue(rain)
                .then(0.01)
                .orElse(0.4)
        );

        //The grass being wet is dependent on whether or not it rained or
        //the sprinkler was on.
        BoolVertex wetGrass = new BernoulliVertex(
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
        long keepSampleCount = 100000;
        NetworkSamplesGenerator networkSamplesGenerator = MetropolisHastings.withDefaultConfig().generatePosteriorSamples(
            new BayesianNetwork(wetGrass.getConnectedGraph()),
            Arrays.asList(sprinkler, rain)
        ).dropCount(10000).downSampleInterval(2);

        double probabilityOfRainGivenWetGrass = networkSamplesGenerator.stream()
            .limit(keepSampleCount)
            .filter(isRaining -> isRaining.get(rain).scalar())
            .count() / (double) keepSampleCount;

        System.out.println();
        System.out.println("Probability Of Rain Given Wet Grass: " + probabilityOfRainGivenWetGrass);
    }
}
