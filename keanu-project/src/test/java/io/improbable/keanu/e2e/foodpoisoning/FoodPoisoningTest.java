package io.improbable.keanu.e2e.foodpoisoning;


import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.network.BayesNet;
import io.improbable.keanu.plating.Plate;
import io.improbable.keanu.plating.PlateBuilder;
import io.improbable.keanu.plating.Plates;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.probabilistic.Flip;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.DoubleUnaryOpLambda;
import org.junit.Before;
import org.junit.Test;

import java.util.Random;
import java.util.function.Consumer;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

public class FoodPoisoningTest {

    private Random random;
    private Flip infectedOysters;
    private Flip infectedLamb;
    private Flip infectedToilet;

    @Before
    public void setup() {
        random = new Random(1);
        infectedOysters = new Flip(0.4);
        infectedLamb = new Flip(0.4);
        infectedToilet = new Flip(0.1);
    }

    @Test
    public void oystersAreInfected() {
        generateSurveyData(50, true, false, false);

        int dropCount = 10000;
        NetworkSamples samples = sample(15000).drop(dropCount);

        assertEquals(1.0, samples.get(infectedOysters).probability(v -> v), 0.01);
        assertEquals(0.0, samples.get(infectedLamb).probability(v -> v), 0.01);
        assertEquals(0.0, samples.get(infectedToilet).probability(v -> v), 0.01);
    }

    @Test
    public void lambAndOystersAreInfected() {
        generateSurveyData(50, true, true, false);
        NetworkSamples samples = sample(15000);

        int dropCount = 10000;
        assertEquals(1.0, samples.get(infectedOysters).probability(v -> v), 0.01);
        assertEquals(1.0, samples.get(infectedLamb).probability(v -> v), 0.01);
        assertEquals(0.0, samples.get(infectedToilet).probability(v -> v), 0.01);
    }

    @Test
    public void nothingIsInfected() {
        generateSurveyData(50, false, false, false);
        NetworkSamples samples = sample(15000);

        int dropCount = 10000;
        assertEquals(0.0, samples.get(infectedOysters).probability(v -> v), 0.01);
        assertEquals(0.0, samples.get(infectedLamb).probability(v -> v), 0.01);
        assertEquals(0.0, samples.get(infectedToilet).probability(v -> v), 0.01);
    }

    public NetworkSamples sample(int n) {
        BayesNet myNet = new BayesNet(infectedOysters.getConnectedGraph());
        myNet.probeForNonZeroMasterP(100, random);
        assertNotEquals(Double.NEGATIVE_INFINITY, myNet.getLogOfMasterP());
        return MetropolisHastings.getPosteriorSamples(myNet, myNet.getLatentVertices(), n, random);
    }

    public void generateSurveyData(int peopleCount, boolean oystersAreInfected, boolean lambIsInfected, boolean toiletIsInfected) {

        Consumer<Plate> personMaker = (plate) -> {
            Flip didEatOysters = plate.add("didEatOysters", new Flip(0.4));
            Flip didEatLamb = plate.add("didEatLamb", new Flip(0.4));
            Flip didEatPoo = plate.add("didEatPoo", new Flip(0.4));

            BoolVertex ingestedPathogen =
                    didEatOysters.and(infectedOysters).or(
                            didEatLamb.and(infectedLamb).or(
                                    didEatPoo.and(infectedToilet)
                            )
                    );

            DoubleUnaryOpLambda<Boolean> pIll = plate.add("pIll", new DoubleUnaryOpLambda<>(ingestedPathogen, (i) -> i ? 0.9 : 0.01));

            plate.add("isIll", new Flip(pIll));
        };

        Plates personPlates = new PlateBuilder()
                .count(peopleCount)
                .withFactory(personMaker)
                .build();

        infectedOysters.observe(oystersAreInfected);
        infectedLamb.observe(lambIsInfected);
        infectedToilet.observe(toiletIsInfected);

        sample(10000);

        personPlates.asList().forEach(plate -> {
            plate.get("didEatOysters").observeOwnValue();
            plate.get("didEatLamb").observeOwnValue();
            plate.get("didEatPoo").observeOwnValue();
            plate.get("isIll").observeOwnValue();
        });

        infectedOysters.unobserve();
        infectedLamb.unobserve();
        infectedToilet.unobserve();
    }

}