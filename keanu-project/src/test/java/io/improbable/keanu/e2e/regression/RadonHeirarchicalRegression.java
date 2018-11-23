package io.improbable.keanu.e2e.regression;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.algorithms.mcmc.NUTS;
import io.improbable.keanu.algorithms.mcmc.proposal.GaussianProposalDistribution;
import io.improbable.keanu.algorithms.mcmc.proposal.ProposalDistribution;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizer;
import io.improbable.keanu.model.regression.RegressionModel;
import io.improbable.keanu.model.regression.RegressionRegularization;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.util.csv.ReadCsv;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.HalfCauchyVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.HalfGaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import umontreal.ssj.probdist.HalfNormalDist;

public class RadonHeirarchicalRegression {

    @Rule
    public DeterministicRule rule = new DeterministicRule();

    private List<Data> radonData;

    @Before
    public void loadCsv() {
        radonData = ReadCsv.fromResources("data/datasets/radon/radon.csv")
            .asRowsDefinedBy(Data.class)
            .load(true);
    }

    @Test
    public void linearRegressionRadon() {
        RegressionModel model = buildSimpleNetwork(radonData);
        Assert.assertTrue(model.getWeight(0) > -0.7 && model.getWeight(0) < -0.4);
        Assert.assertTrue(model.getIntercept() > 1.2 && model.getIntercept() < 1.5);
    }

    @Test
    public void linearRegressionWithOneHeirarchy() {
        buildSingleHeirarchicalNetwork(radonData);
    }

    @Test
    public void heirarchicalLinearRegressionRadon() {
        Map<String, List<Data>> countryRadon = partitionDataOnCounty(radonData, 1);

        BayesianNetwork bayesianNetwork = buildHeirarchicalNetwork(countryRadon);

//        sample(bayesianNetwork, muAlpha, muBeta);
//        NUTSSample(bayesianNetwork, muAlpha, muBeta);
        optimise(bayesianNetwork);
    }

    private Map<String, List<Data>> partitionDataOnCounty(List<Data> radonData, int countyCount) {
        Map<String, List<Data>> countyRadon = new HashMap<>();
        for (Data data : radonData) {
            if (countyRadon.keySet().size() == countyCount && !countyRadon.containsKey(data.county)) {
                break;
            }
            countyRadon.computeIfAbsent(data.county, k -> new ArrayList<>()).add(data);
        }
        return countyRadon;
    }

    private RegressionModel buildSimpleNetwork(List<Data> data) {
        double[] radon = data.stream().mapToDouble(k -> k.log_radon).toArray();
        double[] floor = data.stream().mapToDouble(k -> k.floor).toArray();
        DoubleTensor y = DoubleTensor.create(radon);
        DoubleTensor x = DoubleTensor.create(floor);
        RegressionModel model = RegressionModel.
            withTrainingData(x, y).
            withRegularization(RegressionRegularization.RIDGE).
            withPriorOnWeights(ConstantVertex.of(0.), ConstantVertex.of(5.)).
            withPriorOnIntercept(ConstantVertex.of(0.), ConstantVertex.of(5.)).
            build();
        model.observe();
        model.fit();
        System.out.println("Running linear regression");
        return model;
    }

    private RegressionModel buildSingleHeirarchicalNetwork(List<Data> radonData) {
        GaussianVertex muAlpha = new GaussianVertex(0, 5).setLabel("MuAlpha");
        GaussianVertex muBeta = new GaussianVertex(0, 5).setLabel("MuBeta");

        HalfGaussianVertex sigmaAlpha = new HalfGaussianVertex(100.).setLabel("SigmaAlpha");
        HalfGaussianVertex sigmaBeta = new HalfGaussianVertex(100.).setLabel("SigmaBeta");

        double[] floor = radonData.stream().mapToDouble(k -> k.floor).toArray();
        double[] radon = radonData.stream().mapToDouble(k -> k.log_radon).toArray();
        DoubleTensor x = DoubleTensor.create(floor);
        DoubleTensor y = DoubleTensor.create(radon);
        RegressionModel model = RegressionModel.
            withTrainingData(x, y).
            withRegularization(RegressionRegularization.RIDGE).
            withPriorOnWeights(muBeta, sigmaBeta).
            withPriorOnIntercept(muAlpha, sigmaAlpha).
            build();
        model.observe();

        optimise(new BayesianNetwork(muAlpha.getConnectedGraph()));
        System.out.println("Model weights");
        System.out.println(model.getWeights());
        System.out.println("Model intercept");
        System.out.println(model.getIntercept());
        return model;
    }

    private BayesianNetwork buildHeirarchicalNetwork(Map<String, List<Data>> countryRadon) {
        GaussianVertex muAlpha = new GaussianVertex(0, 100).setLabel("MuAlpha");
        GaussianVertex muBeta = new GaussianVertex(0, 100).setLabel("MuBeta");

        HalfGaussianVertex sigmaAlpha = new HalfGaussianVertex(5.).setLabel("SigmaAlpha");
        HalfGaussianVertex sigmaBeta = new HalfGaussianVertex(5.).setLabel("SigmaBeta");

        for (String county : countryRadon.keySet()) {
            double[] radon = countryRadon.get(county).stream().mapToDouble(k -> k.log_radon).toArray();
            double[] floor = countryRadon.get(county).stream().mapToDouble(k -> k.floor).toArray();
            DoubleTensor x = DoubleTensor.create(floor);
            DoubleTensor y = DoubleTensor.create(radon);
            RegressionModel model = RegressionModel.
                withTrainingData(x, y).
                withRegularization(RegressionRegularization.RIDGE).
                withPriorOnWeights(muBeta, sigmaBeta).
                withPriorOnIntercept(muAlpha, sigmaAlpha).
                build();
            model.observe();
            model.fit();
        }

        muAlpha.setValue(1.5);
        muBeta.setValue(-0.7);
        sigmaAlpha.setValue(2);
        sigmaBeta.setValue(2);
        return new BayesianNetwork(muAlpha.getConnectedGraph());
    }

    private void optimise(BayesianNetwork bayesianNetwork) {
        Vertex muAlpha = bayesianNetwork.getVertexByLabel(new VertexLabel("MuAlpha"));
        Vertex muBeta = bayesianNetwork.getVertexByLabel(new VertexLabel("MuBeta"));
        Vertex sigmaAlpha = bayesianNetwork.getVertexByLabel(new VertexLabel("SigmaAlpha"));
        Vertex sigmaBeta = bayesianNetwork.getVertexByLabel(new VertexLabel("SigmaBeta"));

        GradientOptimizer optimizer = GradientOptimizer.builder().bayesianNetwork(bayesianNetwork).maxEvaluations(10000).build();
        optimizer.maxAPosteriori();

        System.out.println("MuA: " + muAlpha.getValue());
        System.out.println("MuB: " + muBeta.getValue());
        System.out.println("SiA: " + sigmaAlpha.getValue());
        System.out.println("SiB: " + sigmaBeta.getValue());
    }

    private void sample(BayesianNetwork bayesianNetwork) {
        Vertex muAlpha = bayesianNetwork.getVertexByLabel(new VertexLabel("MuAlpha"));
        Vertex muBeta = bayesianNetwork.getVertexByLabel(new VertexLabel("MuBeta"));

        NetworkSamples posteriorSamples = MetropolisHastings.builder().build().
            getPosteriorSamples(
                bayesianNetwork,
                Arrays.asList(muAlpha, muBeta),
                50000
            ).drop(5000)
            .downSample(bayesianNetwork.getContinuousLatentVertices().size());

        double averagePosteriorA = posteriorSamples.getDoubleTensorSamples(muAlpha).getAverages().scalar();
        double averagePosteriorB = posteriorSamples.getDoubleTensorSamples(muBeta).getAverages().scalar();

        System.out.println("Alpha: " + averagePosteriorA);
        System.out.println("Beta: " + averagePosteriorB);
    }


    private void NUTSSample(BayesianNetwork bayesianNetwork) {
        Vertex muAlpha = bayesianNetwork.getVertexByLabel(new VertexLabel("MuAlpha"));
        Vertex muBeta = bayesianNetwork.getVertexByLabel(new VertexLabel("MuBeta"));

        NetworkSamples posteriorSamples = NUTS.builder()
            .maxTreeHeight(5)
            .build()
            .getPosteriorSamples(bayesianNetwork, Arrays.asList(muAlpha, muBeta), 500).downSample(100);

        double averagePosteriorA = posteriorSamples.getDoubleTensorSamples(muAlpha).getAverages().scalar();
        double averagePosteriorB = posteriorSamples.getDoubleTensorSamples(muBeta).getAverages().scalar();

        System.out.println("Alpha: " + averagePosteriorA);
        System.out.println("Beta: " + averagePosteriorB);
    }

    public static class Data {
        public String county;
        public double log_radon;
        public double floor;
    }

}
