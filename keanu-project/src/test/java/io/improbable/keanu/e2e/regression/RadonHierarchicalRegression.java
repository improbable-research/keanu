package io.improbable.keanu.e2e.regression;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.mcmc.nuts.NUTS;
import io.improbable.keanu.model.regression.RegressionModel;
import io.improbable.keanu.model.regression.RegressionRegularization;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.KeanuProbabilisticModelWithGradient;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.util.csv.ReadCsv;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.HalfGaussianVertex;
import lombok.Value;
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

/*
 * Implementation of https://docs.pymc.io/notebooks/GLM-hierarchical.html
 */
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
        assertLinearRegressionIsCorrect(model);
    }

    @Test
    public void canPerformRegressionWithEightHierarchies() {
        buildAndRunHierarchicalNetwork(radonData, 8);
    }

    private RegressionModel linearRegression(Map<String, List<Data>> data) {
        // Build one non-hierarchical model combining all counties' data
        double[] radon = data.values().stream().flatMapToDouble(l -> l.stream().mapToDouble(k -> k.log_radon)).toArray();
        double[] floor = data.values().stream().flatMapToDouble(l -> l.stream().mapToDouble(k -> k.floor)).toArray();
        DoubleTensor y = DoubleTensor.create(radon, radon.length, 1);
        DoubleTensor x = DoubleTensor.create(floor, floor.length, 1);

        return RegressionModel.
            withTrainingData(x, y).
            withRegularization(RegressionRegularization.RIDGE).
            withPriorOnWeights(0., 5.).
            withPriorOnIntercept(0., 5.).
            build();
    }

    public void buildAndRunHierarchicalNetwork(Map<String, List<Data>> radonData, int numberOfModels) {

        if (numberOfModels > radonData.size()) {
            throw new IllegalArgumentException("Not enough data for " + numberOfModels + " models!");
        }

        GaussianVertex muAlpha = new GaussianVertex(0, 10).setLabel("MuIntercept");
        GaussianVertex muBeta = new GaussianVertex(0, 10).setLabel("MuGradient");

        DoubleVertex sigmaAlpha = new HalfGaussianVertex(0.5).setLabel("SigmaIntercept");
        DoubleVertex sigmaBeta = new HalfGaussianVertex(0.5).setLabel("SigmaGradient");

//        DoubleVertex sigmaAlpha = new GaussianVertex(0, 0.5).pow(2).pow(0.5).setLabel("SigmaIntercept");
//        DoubleVertex sigmaBeta = new GaussianVertex(0, 0.5).pow(2).pow(0.5).setLabel("SigmaGradient");

        final List<SubModel> models = radonData.keySet().stream()
            .sorted()
            .limit(numberOfModels)
            .map(county -> createSubModel(radonData.get(county), muBeta, muAlpha, sigmaBeta, sigmaAlpha))
            .collect(Collectors.toList());

        BayesianNetwork net = new BayesianNetwork(muAlpha.getConnectedGraph());

        final NetworkSamples samples = sampleWithNUTS(net, Arrays.asList(muAlpha, muBeta, sigmaAlpha, sigmaBeta));

        assertSampleAveragesAreCorrect(muAlpha, muBeta, sigmaAlpha, sigmaBeta, models, samples);
    }

    private SubModel createSubModel(List<Data> data,
                                    DoubleVertex muGradient,
                                    DoubleVertex muIntercept,
                                    DoubleVertex sigmaGradient,
                                    DoubleVertex sigmaIntercept) {

        double[] floorForSubModel = data.stream().mapToDouble(d -> d.floor).toArray();
        double[] radonForSubModel = data.stream().mapToDouble(d -> d.log_radon).toArray();

        DoubleVertex x = ConstantVertex.of(DoubleTensor.create(floorForSubModel, floorForSubModel.length, 1));

        DoubleVertex gradient = new GaussianVertex(muGradient, sigmaGradient);
        DoubleVertex intercept = new GaussianVertex(muIntercept, sigmaIntercept);

        DoubleVertex y = x.times(gradient).plus(intercept);

        DoubleVertex yObs = new GaussianVertex(y, 1);
        yObs.observe(DoubleTensor.create(radonForSubModel, floorForSubModel.length, 1));

        return new SubModel(gradient, intercept);
    }

    private NetworkSamples sampleWithNUTS(BayesianNetwork bayesianNetwork, List<Vertex> sampleFrom) {

        // note that way too few samples are taken due to time constraints
        KeanuProbabilisticModelWithGradient probabilisticModel = new KeanuProbabilisticModelWithGradient(bayesianNetwork);
        return NUTS.builder()
            .maxTreeHeight(8)
            .adaptCount(1000)
            .build()
            .getPosteriorSamples(probabilisticModel, sampleFrom, 1000)
            .drop(100);
    }

    private void assertSampleAveragesAreCorrect(final DoubleVertex muAlpha,
                                                final DoubleVertex muBeta,
                                                final DoubleVertex sigmaAlpha,
                                                final DoubleVertex sigmaBeta,
                                                final List<SubModel> models,
                                                final NetworkSamples samples) {
        double averagePosteriorMuA = samples.getDoubleTensorSamples(muAlpha).getAverages().scalar();
        double averagePosteriorSigmaA = samples.getDoubleTensorSamples(sigmaAlpha).getAverages().scalar();
        double averagePosteriorMuB = samples.getDoubleTensorSamples(muBeta).getAverages().scalar();
        double averagePosteriorSigmaB = samples.getDoubleTensorSamples(sigmaBeta).getAverages().scalar();

        assertThat(averagePosteriorMuB, allOf(greaterThan(-0.9), lessThan(-0.4)));
        assertThat(averagePosteriorMuA, allOf(greaterThan(1.2), lessThan(1.8)));

        assertThat(averagePosteriorSigmaB, allOf(greaterThan(0.), lessThan(0.5)));
        assertThat(averagePosteriorSigmaA, allOf(greaterThan(0.), lessThan(0.5)));

        for (SubModel subModel : models) {
            double weight = subModel.getWeightVertex().getValue().scalar();
            double intercept = subModel.getInterceptVertex().getValue().scalar();
            assertThat(weight, both(greaterThan(-3.0)).and(lessThan(0.)));
            assertThat(intercept, both(greaterThan(0.)).and(lessThan(3.0)));
        }
    }

    private void assertLinearRegressionIsCorrect(final RegressionModel linearRegressionModel) {
        assertThat(linearRegressionModel.getWeightVertex().getValue().scalar(),
            both(greaterThan(-0.7)).and(lessThan(-0.4)));
        assertThat(linearRegressionModel.getInterceptVertex().getValue().scalar(),
            both(greaterThan(1.2)).and(lessThan(1.5)));
    }

    @Value
    public static class SubModel {
        DoubleVertex weightVertex;
        DoubleVertex interceptVertex;
    }

    public static class Data {
        public String county;
        public double log_radon;
        public double floor;
    }
}
