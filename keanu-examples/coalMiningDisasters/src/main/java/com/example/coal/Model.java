package com.example.coal;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.algorithms.variational.optimizer.KeanuProbabilisticModel;
import io.improbable.keanu.algorithms.variational.optimizer.ProbabilisticModel;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.GreaterThanVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.probabilistic.ExponentialVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.If;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;
import io.improbable.keanu.vertices.intgr.probabilistic.PoissonVertex;
import io.improbable.keanu.vertices.intgr.probabilistic.UniformIntVertex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Model {

    private static final Logger log = LoggerFactory.getLogger(Model.class);

    public static void main(String[] args) {

        log.info("Loading data from a csv file");
        Data coalMiningDisasterData = Data.load("coal-mining-disaster-data.csv");

        log.info("Creating model using loaded data");
        Model coalMiningDisastersModel = new Model(coalMiningDisasterData);

        log.info("Running model...");
        coalMiningDisastersModel.run();
        log.info("Run complete");

        int switchYear = coalMiningDisastersModel.results
            .getIntegerTensorSamples(coalMiningDisastersModel.switchpoint)
            .getScalarMode();

        log.info("Switch year found: " + switchYear);
    }

    ExponentialVertex earlyRate;
    ExponentialVertex lateRate;
    UniformIntVertex switchpoint;

    Data data;
    NetworkSamples results;

    public Model(Data data) {
        this.data = data;
        KeanuRandom.setDefaultRandomSeed(1);
    }

    /**
     * Runs the MetropolisHastings algorithm and saves the resulting samples to results
     */
    public void run() {
        ProbabilisticModel model = new KeanuProbabilisticModel(buildBayesianNetwork());
        Integer numSamples = 500;

        results = MetropolisHastings.withDefaultConfig().generatePosteriorSamples(
            model,
            model.getLatentVariables()
        ).dropCount(numSamples/5).downSampleInterval(3).generate(numSamples);
    }

    private BayesianNetwork buildBayesianNetwork() {

        switchpoint = new UniformIntVertex(data.startYear, data.endYear + 1);
        earlyRate = new ExponentialVertex(1.0);
        lateRate = new ExponentialVertex(1.0);

        ConstantIntegerVertex years = ConstantVertex.of(data.years);

        DoubleVertex rateForYear = If.isTrue(new GreaterThanVertex<>(switchpoint, years))
            .then(earlyRate)
            .orElse(lateRate);

        PoissonVertex disastersForYear = new PoissonVertex(rateForYear);

        disastersForYear.observe(data.disasters);

        return new BayesianNetwork(switchpoint.getConnectedGraph());
    }

}
