package io.improbable.keanu.e2e.foodpoisoning;


import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.KeanuProbabilisticModel;
import io.improbable.keanu.plating.Plate;
import io.improbable.keanu.plating.PlateBuilder;
import io.improbable.keanu.plating.Plates;
import io.improbable.keanu.testcategory.Slow;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.If;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.experimental.categories.Category;

import java.util.function.Consumer;

import static io.improbable.keanu.Keanu.Sampling.MetropolisHastings;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

public class FoodPoisoningTest {

    private KeanuRandom random;
    private BernoulliVertex infectedOysters;
    private BernoulliVertex infectedLamb;
    private BernoulliVertex infectedToilet;

    @Rule
    public DeterministicRule rule = new DeterministicRule();

    @Before
    public void setup() {
        random = new KeanuRandom(1);
        infectedOysters = new BernoulliVertex(0.4);
        infectedLamb = new BernoulliVertex(0.4);
        infectedToilet = new BernoulliVertex(0.1);
    }

    @Category(Slow.class)
    @Test
    public void oystersAreInfected() {
        generateSurveyData(30, true, false, false);

        int dropCount = 2000;
        NetworkSamples samples = sample(5000).drop(dropCount);

        assertEquals(1.0, samples.get(infectedOysters).probability(v -> v.scalar()), 0.01);
        assertEquals(0.0, samples.get(infectedLamb).probability(v -> v.scalar()), 0.01);
        assertEquals(0.0, samples.get(infectedToilet).probability(v -> v.scalar()), 0.01);
    }

    @Category(Slow.class)
    @Test
    public void lambAndOystersAreInfected() {
        generateSurveyData(30, true, true, false);

        int dropCount = 2000;
        NetworkSamples samples = sample(5000).drop(dropCount);

        assertEquals(1.0, samples.get(infectedOysters).probability(v -> v.scalar()), 0.01);
        assertEquals(1.0, samples.get(infectedLamb).probability(v -> v.scalar()), 0.01);
        assertEquals(0.0, samples.get(infectedToilet).probability(v -> v.scalar()), 0.01);
    }

    @Category(Slow.class)
    @Test
    public void nothingIsInfected() {
        generateSurveyData(30, false, false, false);

        int dropCount = 2000;
        NetworkSamples samples = sample(5000).drop(dropCount);

        assertEquals(0.0, samples.get(infectedOysters).probability(v -> v.scalar()), 0.01);
        assertEquals(0.0, samples.get(infectedLamb).probability(v -> v.scalar()), 0.01);
        assertEquals(0.0, samples.get(infectedToilet).probability(v -> v.scalar()), 0.01);
    }

    public NetworkSamples sample(int n) {
        BayesianNetwork myNet = new BayesianNetwork(infectedOysters.getConnectedGraph());
        myNet.probeForNonZeroProbability(100, random);
        assertNotEquals(Double.NEGATIVE_INFINITY, myNet.getLogOfMasterP());
        KeanuProbabilisticModel model = new KeanuProbabilisticModel(myNet);
        return MetropolisHastings.withDefaultConfig(random).getPosteriorSamples(model, myNet.getLatentVertices(), n);
    }

    public void generateSurveyData(int peopleCount, boolean oystersAreInfected, boolean lambIsInfected, boolean toiletIsInfected) {

        VertexLabel didEatOystersLabel = new VertexLabel("didEatOysters");
        VertexLabel didEatLambLabel = new VertexLabel("didEatLamb");
        VertexLabel didEatPooLabel = new VertexLabel("didEatPoo");
        VertexLabel isIllLabel = new VertexLabel("isIll");
        VertexLabel pIllLabel = new VertexLabel("pIll");

        Consumer<Plate> personMaker = (plate) -> {
            BernoulliVertex didEatOysters = plate.add(didEatOystersLabel, new BernoulliVertex(0.4));
            BernoulliVertex didEatLamb = plate.add(didEatLambLabel, new BernoulliVertex(0.4));
            BernoulliVertex didEatPoo = plate.add(didEatPooLabel, new BernoulliVertex(0.4));

            BooleanVertex ingestedPathogen =
                didEatOysters.and(infectedOysters).or(
                    didEatLamb.and(infectedLamb).or(
                        didEatPoo.and(infectedToilet)
                    )
                );

            DoubleVertex pIll = If.isTrue(ingestedPathogen)
                .then(0.9)
                .orElse(0.1);

            plate.add(pIllLabel, pIll);
            plate.add(isIllLabel, new BernoulliVertex(pIll));
        };

        Plates personPlates = new PlateBuilder()
            .count(peopleCount)
            .withFactory(personMaker)
            .build();

        infectedOysters.observe(oystersAreInfected);
        infectedLamb.observe(lambIsInfected);
        infectedToilet.observe(toiletIsInfected);

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