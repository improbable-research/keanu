package io.improbable.docs;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.probabilistic.Flip;
import io.improbable.keanu.vertices.generic.nonprobabilistic.CPT;
import io.improbable.keanu.vertices.generic.nonprobabilistic.If;

import java.util.Arrays;

public class WetGrass {

    public static void main(String[] args) {

        BoolVertex rain = new Flip(0.2);

        BoolVertex sprinkler = new Flip(
            If.isTrue(rain)
                .then(0.01)
                .orElse(0.4)
        );

        BoolVertex wetGrass = new Flip(
            CPT.of(sprinkler, rain)
                .when(false, false).then(1e-2)
                .when(false, true).then(0.8)
                .when(true, false).then(0.9)
                .orDefault(0.99)
        );

        wetGrass.observe(true);
        
        NetworkSamples posteriorSamples = MetropolisHastings.getPosteriorSamples(
            new BayesianNetwork(wetGrass.getConnectedGraph()),
            Arrays.asList(sprinkler, rain),
            100000
        ).drop(10000).downSample(2);

        double probabilityOfRainGivenWetGrass = posteriorSamples.get(rain).probability(isRaining -> isRaining.scalar() == true);

        System.out.println(probabilityOfRainGivenWetGrass);
    }
}
