package com.example.coal;


import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.network.BayesNet;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.GreaterThanVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.CastDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.ExponentialVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.generic.nonprobabilistic.ConstantVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.IfVertex;
import io.improbable.keanu.vertices.intgr.probabilistic.PoissonVertex;
import io.improbable.keanu.vertices.intgr.probabilistic.UniformIntVertex;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class Model {

    public static void main(String[] args) {

        System.out.println("Loading data from a csv file");
        Data coalMiningDisasterData = Data.load("coal-mining-disaster-data.csv");

        System.out.println("Creating model using loaded data");
        Model coalMiningDisastersModel = new Model(coalMiningDisasterData);

        System.out.println("Running model...");
        coalMiningDisastersModel.run();
        System.out.println("Run complete");

        int switchYear = coalMiningDisastersModel.results.get(coalMiningDisastersModel.switchpoint).getMode();

        System.out.println("Switch year found: " + switchYear);
    }

    private final KeanuRandom random;

    final ConstantVertex<Integer> startYearVertex;
    final ConstantVertex<Integer> endYearVertex;
    final ExponentialVertex earlyRate;
    final ExponentialVertex lateRate;
    final List<PoissonVertex> disasters;
    final UniformIntVertex switchpoint;

    final Data data;
    NetworkSamples results;

    public Model(Data data) {
        this.data = data;
        random = new KeanuRandom(1);

        startYearVertex = new ConstantVertex<>(data.startYear);
        endYearVertex = new ConstantVertex<>(data.endYear + 1);
        switchpoint = new UniformIntVertex(startYearVertex, endYearVertex);
        earlyRate = new ExponentialVertex(1.0, 1.0);
        lateRate = new ExponentialVertex(1.0, 1.0);

        Stream<IfVertex<Double>> rates = IntStream.range(data.startYear, data.endYear).boxed()
            .map(ConstantVertex::new)
            .map(year -> {
                GreaterThanVertex<Integer, Integer> switchpointGreaterThanYear = new GreaterThanVertex<>(
                    switchpoint,
                    year
                );
                return new IfVertex<>(switchpointGreaterThanYear, earlyRate, lateRate);
            });

        disasters = rates
            .map(CastDoubleVertex::new)
            .map(PoissonVertex::new)
            .collect(Collectors.toList());

        IntStream.range(0, disasters.size()).forEach(i -> {
            Integer year = data.startYear + i;
            Integer observedValue = data.yearToDisasterCounts.get(year);
            disasters.get(i).observe(observedValue);
        });
    }

    /**
     * Runs the MetropolisHastings algorithm and saves the resulting samples to results
     */
    public void run() {
        BayesNet net = new BayesNet(switchpoint.getConnectedGraph());
        Integer numSamples = 50000;
        NetworkSamples posteriorDistSamples = MetropolisHastings.getPosteriorSamples(net, net.getLatentVertices(), numSamples, random);

        Integer dropCount = 1000;
        results = posteriorDistSamples.drop(dropCount).downSample(5);
    }

}
