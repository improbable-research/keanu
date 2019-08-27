package io.improbable.keanu.e2e.regression;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.mcmc.nuts.NUTS;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.KeanuProbabilisticModelWithGradient;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.testcategory.Slow;
import io.improbable.keanu.util.csv.ReadCsv;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.HalfCauchyVertex;
import lombok.Value;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Rule;
import org.junit.Test;
import org.junit.experimental.categories.Category;

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
    @Category(Slow.class)
    @Ignore("Ignoring this until we can run a large enough version of this quickly enough to be useful")
    public void canPerformRegressionWithEightHierarchies() {
        buildAndRunHierarchicalNetwork(radonData, 8);
    }

    public void buildAndRunHierarchicalNetwork(Map<String, List<Data>> radonData, int numberOfModels) {

        if (numberOfModels > radonData.size()) {
            throw new IllegalArgumentException("Not enough data for " + numberOfModels + " models!");
        }

        DoubleVertex muIntercept = new GaussianVertex(0, 1.0).setLabel("MuIntercept");
        DoubleVertex muGradient = new GaussianVertex(0, 1.0).setLabel("MuGradient");

        DoubleVertex sigmaIntercept = new HalfCauchyVertex(1.0).setLabel("SigmaIntercept");
        DoubleVertex sigmaGradient = new HalfCauchyVertex(1.0).setLabel("SigmaGradient");

        DoubleVertex eps = new HalfCauchyVertex(1.0).setLabel("Eps");

        final List<SubModel> models = radonData.keySet().stream()
            .sorted()
            .limit(numberOfModels)
            .map(county -> createSubModel(radonData.get(county), muGradient, muIntercept, sigmaGradient, sigmaIntercept, eps))
            .collect(Collectors.toList());

        BayesianNetwork net = new BayesianNetwork(muIntercept.getConnectedGraph());

        final NetworkSamples samples = sampleWithNUTS(net, Arrays.asList(muIntercept, muGradient, sigmaIntercept, sigmaGradient, eps));

        assertSampleAveragesAreCorrect(muIntercept, muGradient, sigmaIntercept, sigmaGradient, eps, models, samples);
    }

    private SubModel createSubModel(List<Data> data,
                                    DoubleVertex muGradient,
                                    DoubleVertex muIntercept,
                                    DoubleVertex sigmaGradient,
                                    DoubleVertex sigmaIntercept,
                                    DoubleVertex eps) {

        double[] floorForSubModel = data.stream().mapToDouble(d -> d.floor).toArray();
        double[] radonForSubModel = data.stream().mapToDouble(d -> d.log_radon).toArray();

        DoubleVertex x = ConstantVertex.of(DoubleTensor.create(floorForSubModel, floorForSubModel.length));

        DoubleVertex gradient = new GaussianVertex(muGradient, sigmaGradient);
        DoubleVertex intercept = new GaussianVertex(muIntercept, sigmaIntercept);

        DoubleVertex y = x.times(gradient).plus(intercept);

        DoubleVertex yObs = new GaussianVertex(y, eps);
        yObs.observe(DoubleTensor.create(radonForSubModel, floorForSubModel.length));

        return new SubModel(gradient, intercept);
    }

    private NetworkSamples sampleWithNUTS(BayesianNetwork bayesianNetwork, List<Vertex> sampleFrom) {

        int sampleCount = 1000;
        KeanuProbabilisticModelWithGradient probabilisticModel = new KeanuProbabilisticModelWithGradient(bayesianNetwork);
        return NUTS.builder()
            .adaptCount(sampleCount)
            .maxTreeHeight(8)
            .build()
            .getPosteriorSamples(probabilisticModel, sampleFrom, sampleCount)
            .drop(sampleCount / 4);
    }

    private void assertSampleAveragesAreCorrect(final DoubleVertex muIntercept,
                                                final DoubleVertex muGradient,
                                                final DoubleVertex sigmaIntercept,
                                                final DoubleVertex sigmaGradient,
                                                final DoubleVertex eps,
                                                final List<SubModel> models,
                                                final NetworkSamples samples) {

        double averagePosteriorMuIntercept = samples.getDoubleTensorSamples(muIntercept).getAverages().scalar();
        double averagePosteriorSigmaIntercept = samples.getDoubleTensorSamples(sigmaIntercept).getAverages().scalar();
        double averagePosteriorMuGradient = samples.getDoubleTensorSamples(muGradient).getAverages().scalar();
        double averagePosteriorSigmaGradient = samples.getDoubleTensorSamples(sigmaGradient).getAverages().scalar();
        double averagePosteriorEps = samples.getDoubleTensorSamples(eps).getAverages().scalar();

        //-0.65
        assertThat(averagePosteriorMuGradient, allOf(greaterThan(-0.9), lessThan(-0.4)));

        //1.49
        assertThat(averagePosteriorMuIntercept, allOf(greaterThan(1.2), lessThan(1.8)));

        //0.3
        assertThat(averagePosteriorSigmaGradient, allOf(greaterThan(0.), lessThan(0.5)));

        //0.325
        assertThat(averagePosteriorSigmaIntercept, allOf(greaterThan(0.), lessThan(0.5)));

        //0.72
        assertThat(averagePosteriorEps, allOf(greaterThan(0.6), lessThan(0.8)));

        for (SubModel subModel : models) {
            double weight = subModel.getWeightVertex().getValue().scalar();
            double intercept = subModel.getInterceptVertex().getValue().scalar();
            assertThat(weight, both(greaterThan(-3.0)).and(lessThan(1.0)));
            assertThat(intercept, both(greaterThan(0.)).and(lessThan(3.0)));
        }
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
