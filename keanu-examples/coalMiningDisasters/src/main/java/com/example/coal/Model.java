package com.example.coal;


import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.network.BayesNet;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.GreaterThanVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.CastDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.ExponentialVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.ConstantVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.IfVertex;
import io.improbable.keanu.vertices.intgr.probabilistic.PoissonVertex;
import io.improbable.keanu.vertices.intgr.probabilistic.UniformIntVertex;

import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class Model {

    public static void main(String[] args) {

        //Load data from a csv file
        Data coalMiningDisasterData = Data.load("coal-mining-disaster-data.csv");

        //create my model using the data
        Model coalMiningDisastersModel = new Model(coalMiningDisasterData);
        coalMiningDisastersModel.run();
    }

    private final Random r;

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
        r = new Random(1);

        startYearVertex = new ConstantVertex<>(data.startYear);
        endYearVertex = new ConstantVertex<>(data.endYear + 1);
        switchpoint = new UniformIntVertex(startYearVertex, endYearVertex, r);
        earlyRate = new ExponentialVertex(new ConstantDoubleVertex(1.0), new ConstantDoubleVertex(1.0), r);
        lateRate = new ExponentialVertex(new ConstantDoubleVertex(1.0), new ConstantDoubleVertex(1.0), r);

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
                .map(rate -> new PoissonVertex(rate, r))
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
        NetworkSamples posteriorDistSamples = MetropolisHastings.getPosteriorSamples(net, net.getLatentVertices(), numSamples, r);

        Integer dropCount = 1000;
        results = posteriorDistSamples.drop(dropCount).downSample(5);
    }

}
