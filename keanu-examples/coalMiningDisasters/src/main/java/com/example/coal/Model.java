package com.example.coal;


import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.booltensor.nonprobabilistic.operators.binary.compare.GreaterThanVertex;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.dbltensor.probabilistic.TensorExponentialVertex;
import io.improbable.keanu.vertices.generictensor.nonprobabilistic.If;
import io.improbable.keanu.vertices.intgrtensor.nonprobabilistic.ConstantIntegerVertex;
import io.improbable.keanu.vertices.intgrtensor.probabilistic.PoissonVertex;
import io.improbable.keanu.vertices.intgrtensor.probabilistic.UniformIntVertex;

public class Model {

    public static void main(String[] args) {

        System.out.println("Loading data from a csv file");
        Data coalMiningDisasterData = Data.load("coal-mining-disaster-data.csv");

        System.out.println("Creating model using loaded data");
        Model coalMiningDisastersModel = new Model(coalMiningDisasterData);

        System.out.println("Running model...");
        coalMiningDisastersModel.run();
        System.out.println("Run complete");

        int switchYear = coalMiningDisastersModel.results
            .getIntegerTensorSamples(coalMiningDisastersModel.switchpoint)
            .getScalarMode();

        System.out.println("Switch year found: " + switchYear);
    }

    TensorExponentialVertex earlyRate;
    TensorExponentialVertex lateRate;
    UniformIntVertex switchpoint;

    KeanuRandom random;
    Data data;
    NetworkSamples results;

    public Model(Data data) {
        this.data = data;
        random = new KeanuRandom(1);
    }

    /**
     * Runs the MetropolisHastings algorithm and saves the resulting samples to results
     */
    public void run() {
        BayesianNetwork net = buildBayesianNetwork();
        Integer numSamples = 50000;

        NetworkSamples posteriorDistSamples = MetropolisHastings.getPosteriorSamples(
            net,
            net.getLatentVertices(),
            numSamples,
            random
        );

        results = posteriorDistSamples.drop(10000).downSample(3);
    }

    private BayesianNetwork buildBayesianNetwork() {

        switchpoint = new UniformIntVertex(data.startYear, data.endYear + 1);
        earlyRate = new TensorExponentialVertex(1.0, 1.0);
        lateRate = new TensorExponentialVertex(1.0, 1.0);

        ConstantIntegerVertex years = new ConstantIntegerVertex(IntegerTensor.create(data.years));

        DoubleTensorVertex rateForYear = If.isTrue(new GreaterThanVertex<>(switchpoint, years))
            .then(earlyRate)
            .orElse(lateRate);

        PoissonVertex disastersForYear = new PoissonVertex(rateForYear);

        disastersForYear.observe(IntegerTensor.create(data.disasters));

        return new BayesianNetwork(switchpoint.getConnectedGraph());
    }

}
