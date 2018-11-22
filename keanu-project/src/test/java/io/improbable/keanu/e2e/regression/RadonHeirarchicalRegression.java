package io.improbable.keanu.e2e.regression;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.Test;

import io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizer;
import io.improbable.keanu.model.regression.RegressionModel;
import io.improbable.keanu.model.regression.RegressionRegularization;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.util.csv.ReadCsv;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.HalfCauchyVertex;

public class RadonHeirarchicalRegression {

    @Test
    public void doesHeirarchical() {
        List<Data> radonData = ReadCsv.fromResources("data/datasets/radon/radon.csv")
            .asRowsDefinedBy(Data.class)
            .load(true);

        Map<String, List<Data>> countryRadon = new HashMap<>();

        for (Data data : radonData) {
            countryRadon.computeIfAbsent(data.county, k -> new ArrayList<>()).add(data);
        }

        BayesianNetwork bayesianNetwork = buildNetwork(countryRadon);
        bayesianNetwork.probeForNonZeroProbability(500);
        System.out.println("optimising...");
        GradientOptimizer.of(bayesianNetwork).maxAPosteriori();
        System.out.println("optimised");
        Vertex muAlpha = bayesianNetwork.getVertexByLabel(new VertexLabel("MuAlpha"));
        Vertex muBeta = bayesianNetwork.getVertexByLabel(new VertexLabel("MuBeta"));

        System.out.println("Gradient then y");
        System.out.println(muBeta.getValue());
        System.out.println(muAlpha.getValue());
//
//        ProposalDistribution proposalDistribution = new GaussianProposalDistribution(DoubleTensor.scalar(100.));
//
//        NetworkSamples posteriorSamples = MetropolisHastings.builder().proposalDistribution(proposalDistribution).build().
//            getPosteriorSamples(
//            bayesianNetwork,
//            Arrays.asList(muAlpha, muBeta),
//            75000
//        ).drop(5000);
//
//        double averagePosteriorA = posteriorSamples.getDoubleTensorSamples(muAlpha).getAverages().scalar();
//        double averagePosteriorB = posteriorSamples.getDoubleTensorSamples(muBeta).getAverages().scalar();

//        System.out.println("Alpha: " + averagePosteriorA);
//        System.out.println("Beta: " + averagePosteriorB);
    }

    private BayesianNetwork buildNetwork(Map<String, List<Data>> countryRadon) {
        GaussianVertex muAlpha = new GaussianVertex(0, 100).setLabel("MuAlpha");
        GaussianVertex muBeta = new GaussianVertex(0, 100).setLabel("MuBeta");

        HalfCauchyVertex sigmaAlpha = new HalfCauchyVertex(5);
        HalfCauchyVertex sigmaBeta = new HalfCauchyVertex(5);

        for (String county : countryRadon.keySet()) {
            double[] radon = countryRadon.get(county).stream().mapToDouble(k -> k.log_radon).toArray();
            double[] floor = countryRadon.get(county).stream().mapToDouble(k -> k.floor).toArray();
            DoubleTensor x = DoubleTensor.create(floor);
            DoubleTensor y = DoubleTensor.create(radon);
            RegressionModel.
                withTrainingData(x, y).
                withRegularization(RegressionRegularization.RIDGE).
                withPriorOnWeights(muBeta, sigmaBeta).
                withPriorOnIntercept(muAlpha, sigmaAlpha).
                build();
        }

        return new BayesianNetwork(muAlpha.getConnectedGraph());
    }

    public static class Data {
        public String county;
        public double log_radon;
        public double floor;
    }

}
