package io.improbable.keanu.e2e.regression;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.mcmc.nuts.NUTS;
import io.improbable.keanu.algorithms.variational.optimizer.KeanuOptimizer;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizer;
import io.improbable.keanu.model.regression.RegressionModel;
import io.improbable.keanu.model.regression.RegressionRegularization;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.util.csv.ReadCsv;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.HalfGaussianVertex;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.both;
import static org.hamcrest.Matchers.greaterThan;
import static org.hamcrest.Matchers.lessThan;
import static org.hamcrest.core.AllOf.allOf;

public class RadonHierarchicalRegression {

    @Rule
    public DeterministicRule rule = new DeterministicRule();

    private Map<String, List<Data>> radonData;

    @Before
    public void readRadonCSV() {
        radonData = ReadCsv.fromResources("data/datasets/radon/radon.csv")
            .asRowsDefinedBy(Data.class)
            .load(true)
            .stream()
            .collect(Collectors.groupingBy(d -> d.county));
    }

    @Test
    public void canPerformSimpleLinearRegression() {
        RegressionModel model = linearRegression(radonData);
        assertThat(model.getWeightVertex().getValue().scalar(), both(greaterThan(-0.7)).and(lessThan(-0.4)));
        assertThat(model.getInterceptVertex().getValue().scalar(), both(greaterThan(1.2)).and(lessThan(1.5)));
    }

    @Test
    public void canPerformRegressionWithTwentyHierarchies() {
        buildAndRunHierarchicalNetwork(radonData, 20);
    }

    private RegressionModel linearRegression(Map<String, List<Data>> data) {
        double[] radon = data.values().stream().flatMapToDouble(l -> l.stream().mapToDouble(k -> k.log_radon)).toArray();
        double[] floor = data.values().stream().flatMapToDouble(l -> l.stream().mapToDouble(k -> k.floor)).toArray();
        DoubleTensor y = DoubleTensor.create(radon, radon.length, 1);
        DoubleTensor x = DoubleTensor.create(floor, floor.length, 1);

        RegressionModel model = RegressionModel.
            withTrainingData(x, y).
            withRegularization(RegressionRegularization.RIDGE).
            withPriorOnWeights(0., 5.).
            withPriorOnIntercept(0., 5.).
            build();

        return model;
    }

    private void buildAndRunHierarchicalNetwork(Map<String, List<Data>> radonData, int numberOfModels) {
        GaussianVertex muAlpha = new GaussianVertex(new long[]{1, 1}, 0, 100).setLabel("MuIntercept");
        GaussianVertex muBeta = new GaussianVertex(new long[]{1, 1}, 0, 100).setLabel("MuGradient");

        HalfGaussianVertex sigmaAlpha = new HalfGaussianVertex(new long[]{1, 1}, 10.).setLabel("SigmaIntercept");
        HalfGaussianVertex sigmaBeta = new HalfGaussianVertex(new long[]{1, 1}, 10.).setLabel("SigmaGradient");

        List<String> counties = radonData.keySet().stream().sorted().collect(Collectors.toList());

        for (int i = 0; i < Math.min(numberOfModels, counties.size()); i++) {
            List<Data> countyData = radonData.get(counties.get(i));
            createSubModel(countyData, muBeta, muAlpha, sigmaBeta, sigmaAlpha);
        }

        //Set the starting values of the parent random variables
        //This is required for the optimiser to find the correct values of the sub models latents
        muAlpha.setValue(1.);
        sigmaAlpha.setValue(0.5);

        muBeta.setValue(-1.);
        sigmaBeta.setValue(0.5);

        BayesianNetwork net = new BayesianNetwork(muAlpha.getConnectedGraph());
        optimise(net);
        NUTSSample(net);
    }

    private RegressionModel createSubModel(List<Data> data,
                                           DoubleVertex muGradient,
                                           DoubleVertex muIntercept,
                                           DoubleVertex sigmaGradient,
                                           DoubleVertex sigmaIntercept) {
        double[] floorForSubModel = data.stream().mapToDouble(d -> d.floor).toArray();
        double[] radonForSubModel = data.stream().mapToDouble(d -> d.log_radon).toArray();

        DoubleTensor x = DoubleTensor.create(floorForSubModel, floorForSubModel.length, 1);
        DoubleTensor y = DoubleTensor.create(radonForSubModel, floorForSubModel.length, 1);

        RegressionModel model = RegressionModel.
            withTrainingData(x, y).
            withRegularization(RegressionRegularization.RIDGE).
            withPriorOnWeights(muGradient, sigmaGradient).
            withPriorOnIntercept(muIntercept, sigmaIntercept).
            buildWithoutFitting();

        return model;
    }

    private void NUTSSample(BayesianNetwork bayesianNetwork) {
        Vertex muAlpha = bayesianNetwork.getVertexByLabel(new VertexLabel("MuIntercept"));
        Vertex sigmaAlpha = bayesianNetwork.getVertexByLabel(new VertexLabel("SigmaIntercept"));
        Vertex muBeta = bayesianNetwork.getVertexByLabel(new VertexLabel("MuGradient"));
        Vertex sigmaBeta = bayesianNetwork.getVertexByLabel(new VertexLabel("SigmaGradient"));

        NetworkSamples posteriorSamples = NUTS.builder()
            .maxTreeHeight(5)
            .saveStatistics(true)
            .build()
            .getPosteriorSamples(bayesianNetwork, Arrays.asList(muAlpha, muBeta, sigmaAlpha, sigmaBeta), 500).downSample(100);

        double averagePosteriorMuA = posteriorSamples.getDoubleTensorSamples(muAlpha).getAverages().scalar();
        double averagePosteriorSigmaA = posteriorSamples.getDoubleTensorSamples(sigmaAlpha).getAverages().scalar();
        double averagePosteriorMuB = posteriorSamples.getDoubleTensorSamples(muBeta).getAverages().scalar();
        double averagePosteriorSigmaB = posteriorSamples.getDoubleTensorSamples(sigmaBeta).getAverages().scalar();

        assertThat(averagePosteriorMuB, allOf(greaterThan(-0.9), lessThan(-0.4 )));
        assertThat(averagePosteriorMuA, allOf(greaterThan(1.2), lessThan(1.8 )));

        assertThat(averagePosteriorSigmaB, allOf(greaterThan(0.), lessThan(0.5 )));
        assertThat(averagePosteriorSigmaA, allOf(greaterThan(0.), lessThan(0.5 )));

        System.out.println("done");
    }

    private void optimise(BayesianNetwork bayesianNetwork) {
        bayesianNetwork.probeForNonZeroProbability(100);
        GradientOptimizer optimizer = KeanuOptimizer.Gradient.builderFor(bayesianNetwork)
            .absoluteThreshold(0.25)
            .maxEvaluations(10000)
            .build();
        optimizer.maxAPosteriori();
    }

    public static class Data {
        public String county;
        public double log_radon;
        public double floor;
    }
}
