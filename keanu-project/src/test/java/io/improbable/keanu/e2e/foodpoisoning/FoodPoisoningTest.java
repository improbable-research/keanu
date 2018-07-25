package io.improbable.keanu.e2e.foodpoisoning;


import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

import java.util.function.Consumer;

import org.junit.Before;
import org.junit.Test;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.plating.Plate;
import io.improbable.keanu.plating.PlateBuilder;
import io.improbable.keanu.plating.Plates;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.bool.probabilistic.Flip;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.generic.nonprobabilistic.If;

public class FoodPoisoningTest {

    private KeanuRandom random;
    private Flip infectedOysters;
    private Flip infectedLamb;
    private Flip infectedToilet;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
        infectedOysters = new Flip(0.4);
        infectedLamb = new Flip(0.4);
        infectedToilet = new Flip(0.1);
    }

    @Test
    public void oystersAreInfected() {
        generateSurveyData(50, true, false, false);

        int dropCount = 10000;
        NetworkSamples samples = sample(15000).drop(dropCount);

        assertEquals(1.0, samples.get(infectedOysters).probability(v -> v.scalar()), 0.01);
        assertEquals(0.0, samples.get(infectedLamb).probability(v -> v.scalar()), 0.01);
        assertEquals(0.0, samples.get(infectedToilet).probability(v -> v.scalar()), 0.01);
    }

    @Test
    public void lambAndOystersAreInfected() {
        generateSurveyData(50, true, true, false);
        NetworkSamples samples = sample(15000);

        int dropCount = 10000;
        assertEquals(1.0, samples.get(infectedOysters).probability(v -> v.scalar()), 0.01);
        assertEquals(1.0, samples.get(infectedLamb).probability(v -> v.scalar()), 0.01);
        assertEquals(0.0, samples.get(infectedToilet).probability(v -> v.scalar()), 0.01);
    }

    @Test
    public void nothingIsInfected() {
        generateSurveyData(50, false, false, false);
        NetworkSamples samples = sample(15000);

        int dropCount = 10000;
        assertEquals(0.0, samples.get(infectedOysters).probability(v -> v.scalar()), 0.01);
        assertEquals(0.0, samples.get(infectedLamb).probability(v -> v.scalar()), 0.01);
        assertEquals(0.0, samples.get(infectedToilet).probability(v -> v.scalar()), 0.01);
    }

    public NetworkSamples sample(int n) {
        BayesianNetwork myNet = new BayesianNetwork(infectedOysters.getConnectedGraph());
        myNet.probeForNonZeroProbability(100, random);
        assertNotEquals(Double.NEGATIVE_INFINITY, myNet.getLogOfMasterP());
        return MetropolisHastings.getPosteriorSamples(myNet, myNet.getLatentVertices(), n, random);
    }

    public void generateSurveyData(int peopleCount, boolean oystersAreInfected, boolean lambIsInfected, boolean toiletIsInfected) {

        Consumer<Plate> personMaker = (plate) -> {
            Flip didEatOysters = plate.add("didEatOysters", new Flip(0.4));
            Flip didEatLamb = plate.add("didEatLamb", new Flip(0.4));
            Flip didEatPoo = plate.add("didEatPoo", new Flip(0.4));

            BooleanVertex ingestedPathogen =
                didEatOysters.and(infectedOysters).or(
                    didEatLamb.and(infectedLamb).or(
                        didEatPoo.and(infectedToilet)
                    )
                );

            DoubleVertex pIll = If.isTrue(ingestedPathogen)
                .then(0.9)
                .orElse(0.1);

            plate.add("pIll", pIll);
            plate.add("isIll", new Flip(pIll));
        };

        Plates personPlates = new PlateBuilder()
            .count(peopleCount)
            .withFactory(personMaker)
            .build();

        infectedOysters.observe(BooleanTensor.scalar(oystersAreInfected));
        infectedLamb.observe(BooleanTensor.scalar(lambIsInfected));
        infectedToilet.observe(BooleanTensor.scalar(toiletIsInfected));

        sample(10000);

        personPlates.asList().forEach(plate -> {
            for (String name : new String[] {"didEatOysters", "didEatLamb", "didEatPoo", "isIll"}) {
                Vertex<Object> vertex = plate.get(name);
                vertex.observe(vertex.getValue());
            }
        });

        infectedOysters.unobserve();
        infectedLamb.unobserve();
        infectedToilet.unobserve();
    }

}