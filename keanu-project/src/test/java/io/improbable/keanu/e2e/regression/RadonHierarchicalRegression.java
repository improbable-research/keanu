package io.improbable.keanu.e2e.regression;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.algorithms.variational.optimizer.KeanuOptimizer;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.ApacheNonLinearConjugateGradientOptimizer;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizer;
import io.improbable.keanu.model.regression.RegressionModel;
import io.improbable.keanu.model.regression.RegressionRegularization;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.util.csv.ReadCsv;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.HalfGaussianVertex;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.*;

public class RadonHierarchicalRegression {

    @Rule
    public DeterministicRule rule = new DeterministicRule();

    private List<Data> radonData;

    @Before
    public void readRadonCSV() {
        radonData = ReadCsv.fromResources("data/datasets/radon/radon.csv")
            .asRowsDefinedBy(Data.class)
            .load(true);
    }

    @Test
    public void canPerformSimpleLinearRegression() {
        RegressionModel model = linearRegression(radonData);
        assertThat(model.getWeightVertex().getValue().scalar(), both(greaterThan(-0.7)).and(lessThan(-0.4)));
        assertThat(model.getInterceptVertex().getValue().scalar(), both(greaterThan(1.2)).and(lessThan(1.5)));
    }

    @Test
    public void canPerformRegressionWithOneHierarchy() {
        buildAndRunHeirarchicalNetwork(radonData, 1);
    }

    @Test
    public void canPerformRegressionWithTwoHierarchies() {
        buildAndRunHeirarchicalNetwork(radonData, 2);
    }

    @Test
    public void canPerformRegressionWithFourHierarchies() {
        buildAndRunHeirarchicalNetwork(radonData, 4);
    }

    @Test
    public void canPerformRegressionWithTenHierarchies() {
        buildAndRunHeirarchicalNetwork(radonData, 10);
    }

    private RegressionModel linearRegression(List<Data> data) {
        double[] radon = data.stream().mapToDouble(k -> k.log_radon).toArray();
        double[] floor = data.stream().mapToDouble(k -> k.floor).toArray();
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

    private void buildAndRunHeirarchicalNetwork(List<Data> radonData, int numberOfModels) {
        GaussianVertex muAlpha = new GaussianVertex(new long[]{1, 1}, 0, 100).setLabel("MuIntercept");
        GaussianVertex muBeta = new GaussianVertex(new long[]{1, 1}, 0, 100).setLabel("MuGradient");

        HalfGaussianVertex sigmaAlpha = new HalfGaussianVertex(new long[]{1, 1}, 10.).setLabel("SigmaIntercept");
        HalfGaussianVertex sigmaBeta = new HalfGaussianVertex(new long[]{1, 1}, 10.).setLabel("SigmaGradient");

        int numPartitions = radonData.size() / numberOfModels;

        double[] allFloor = radonData.stream().mapToDouble(k -> k.floor).toArray();
        double[] allRadon = radonData.stream().mapToDouble(k -> k.log_radon).toArray();

        List<RegressionModel> models = new ArrayList<>();

        for (int i = 0; i < numberOfModels; i++) {
            RegressionModel model = createSubModel(allFloor, allRadon, i, numPartitions, muBeta, muAlpha, sigmaBeta, sigmaAlpha);
            models.add(model);
        }

        //Set the starting values of the parent random variables
        //This is required for the optimiser to find the correct values of the sub models latents
        muAlpha.setValue(1.);
        sigmaAlpha.setValue(0.5);

        muBeta.setValue(-1.);
        sigmaBeta.setValue(0.5);

        optimise(new BayesianNetwork(muAlpha.getConnectedGraph()), models);
    }

    private RegressionModel createSubModel(double[] allFloor,
                                           double[] allRadon,
                                           int i,
                                           int size,
                                           DoubleVertex muGradient,
                                           DoubleVertex muIntercept,
                                           DoubleVertex sigmaGradient,
                                           DoubleVertex sigmaIntercept) {

        int startIndex = size * i;
        int endIndex = size * (i + 1);

        double[] floorForSubModel = Arrays.copyOfRange(allFloor, startIndex, endIndex);
        double[] radonForSubModel = Arrays.copyOfRange(allRadon, startIndex, endIndex);

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

    private void optimise(BayesianNetwork bayesianNetwork, List<RegressionModel> models) {
        bayesianNetwork.probeForNonZeroProbability(100);
        GradientOptimizer optimizer = KeanuOptimizer.Gradient.builderFor(bayesianNetwork)
            .algorithm(ApacheNonLinearConjugateGradientOptimizer.builder()
                .absoluteThreshold(0.25)
                .maxEvaluations(10000)
                .build()
            ).build();
        optimizer.maxAPosteriori();

        assertValuesAreCorrect(bayesianNetwork, models);
    }

    private void assertValuesAreCorrect(BayesianNetwork bayesianNetwork, List<RegressionModel> models) {
        DoubleVertex muIntercept = (DoubleVertex) bayesianNetwork.getVertexByLabel(new VertexLabel("MuIntercept"));
        DoubleVertex muGradient = (DoubleVertex) bayesianNetwork.getVertexByLabel(new VertexLabel("MuGradient"));
        DoubleVertex sigmaIntercept = (DoubleVertex) bayesianNetwork.getVertexByLabel(new VertexLabel("SigmaIntercept"));
        DoubleVertex sigmaGradient = (DoubleVertex) bayesianNetwork.getVertexByLabel(new VertexLabel("SigmaGradient"));

        Assert.assertTrue(-0.4 > muGradient.getValue().scalar() && muGradient.getValue().scalar() > -0.9);
        Assert.assertTrue(1.8 > muIntercept.getValue().scalar() && muIntercept.getValue().scalar() > 1.2);

        Assert.assertTrue(0.5 > sigmaGradient.getValue().scalar() && sigmaGradient.getValue().scalar() > 0.);
        Assert.assertTrue(0.5 > sigmaIntercept.getValue().scalar() && sigmaIntercept.getValue().scalar() > 0.);

        for (RegressionModel subModel : models) {
            double weight = subModel.getWeightVertex().getValue().scalar();
            double intercept = subModel.getInterceptVertex().getValue().scalar();
            Assert.assertTrue(-0.0 > weight && weight > -1.5);
            Assert.assertTrue(2. > intercept && intercept > 1.);
        }
    }

    public static class Data {
        public String county;
        public double log_radon;
        public double floor;
    }

}
