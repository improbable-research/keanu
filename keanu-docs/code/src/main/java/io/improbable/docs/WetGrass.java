package io.improbable.docs;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.sampling.RejectionSampler;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.probabilistic.Flip;
import io.improbable.keanu.vertices.generic.nonprobabilistic.CPT;
import io.improbable.keanu.vertices.generic.nonprobabilistic.CPTVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.If;

import java.util.Arrays;

public class WetGrass {

    public static void main(String[] args) {

        Flip rain = new Flip(0.2);

        BoolVertex sprinkler = If.isTrue(rain)
            .then(new Flip(0.01))
            .orElse(new Flip(0.4));

        CPTVertex<BooleanTensor> wetGrass = CPT.of(sprinkler, rain)
            .when(false, false).then(new Flip(0.0))
            .when(false, true).then(new Flip(0.8))
            .when(true, false).then(new Flip(0.9))
            .orDefault(new Flip(0.99));

        wetGrass.observe(BooleanTensor.scalar(true));

        NetworkSamples posteriorSamples = RejectionSampler.getPosteriorSamples(
            new BayesianNetwork(wetGrass.getConnectedGraph()),
            Arrays.asList(sprinkler, rain),
            100000
        );

        double probabilityOfRainGivenWetGrass = posteriorSamples.get(rain).probability(isRaining -> isRaining.scalar() == true);

        System.out.println(probabilityOfRainGivenWetGrass);
    }
}
