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
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.generic.nonprobabilistic.If;

public class FoodPoisoningTest {

    private KeanuRandom random;
    private BernoulliVertex infectedOysters;
    private BernoulliVertex infectedLamb;
    private BernoulliVertex infectedToilet;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
        infectedOysters = new BernoulliVertex(0.4);
        infectedLamb = new BernoulliVertex(0.4);
        infectedToilet = new BernoulliVertex(0.1);
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
        return MetropolisHastings.withDefaultConfig(random).getPosteriorSamples(myNet, myNet.getLatentVertices(), n);
    }

    public void generateSurveyData(int peopleCount, boolean oystersAreInfected, boolean lambIsInfected, boolean toiletIsInfected) {

        VertexLabel didEatOystersLabel = new VertexLabel("didEatOysters");
        VertexLabel didEatLambLabel = new VertexLabel("didEatLamb");
        VertexLabel didEatPooLabel = new VertexLabel("didEatPoo");
        VertexLabel isIllLabel = new VertexLabel("isIll");
        VertexLabel pIllLabel = new VertexLabel("pIll");

        Consumer<Plate> personMaker = (plate) -> {
            BernoulliVertex didEatOysters = plate.add( new BernoulliVertex(0.4).labelled(didEatOystersLabel));
            BernoulliVertex didEatLamb = plate.add(new BernoulliVertex(0.4).labelled(didEatLambLabel));
            BernoulliVertex didEatPoo = plate.add(new BernoulliVertex(0.4).labelled(didEatPooLabel));

            BoolVertex ingestedPathogen =
                didEatOysters.and(infectedOysters).or(
                    didEatLamb.and(infectedLamb).or(
                        didEatPoo.and(infectedToilet)
                    )
                );

            DoubleVertex pIll = If.isTrue(ingestedPathogen)
                .then(0.9)
                .orElse(0.1)
                .labelled(pIllLabel);

            plate.add(pIll);
            plate.add(new BernoulliVertex(pIll).labelled(isIllLabel));
        };

        Plates personPlates = new PlateBuilder()
            .count(peopleCount)
            .withFactory(personMaker)
            .build();

        infectedOysters.observe(oystersAreInfected);
        infectedLamb.observe(lambIsInfected);
        infectedToilet.observe(toiletIsInfected);

        sample(10000);

        personPlates.forEach(plate -> {
            plate.get(didEatOystersLabel).observeOwnValue();
            plate.get(didEatLambLabel).observeOwnValue();
            plate.get(didEatPooLabel).observeOwnValue();
            plate.get(isIllLabel).observeOwnValue();
        });

        infectedOysters.unobserve();
        infectedLamb.unobserve();
        infectedToilet.unobserve();
    }

}