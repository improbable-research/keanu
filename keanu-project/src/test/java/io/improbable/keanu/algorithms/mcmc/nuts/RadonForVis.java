package io.improbable.keanu.algorithms.mcmc.nuts;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.KeanuProbabilisticModelWithGradient;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.util.csv.ReadCsv;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class RadonForVis implements ModelForNUTSVis {

    private List<Vertex> toPlot;
    private KeanuProbabilisticModelWithGradient probabilisticModel;

    public RadonForVis() {

        Map<String, List<Data>> radonData = ReadCsv.fromResources("data/datasets/radon/radon.csv")
            .asRowsDefinedBy(Data.class)
            .load(true)
            .stream()
            .collect(Collectors.groupingBy(d -> d.county));


        buildAndRunHierarchicalNetwork(radonData, 8);
    }

    private void buildAndRunHierarchicalNetwork(Map<String, List<Data>> radonData, int numberOfModels) {

        if (numberOfModels > radonData.size()) {
            throw new IllegalArgumentException("Not enough data for " + numberOfModels + " models!");
        }

        GaussianVertex muAlpha = new GaussianVertex(0, 5).setLabel("MuIntercept");
        GaussianVertex muBeta = new GaussianVertex(0, 5).setLabel("MuGradient");

//        HalfGaussianVertex sigmaAlpha = new HalfGaussianVertex(0.5).setLabel("SigmaIntercept");
//        HalfGaussianVertex sigmaBeta = new HalfGaussianVertex(0.5).setLabel("SigmaGradient");

        DoubleVertex sigmaAlpha = new GaussianVertex(0, 0.5).pow(2).pow(0.5).setLabel("SigmaIntercept");
        DoubleVertex sigmaBeta = new GaussianVertex(0, 0.5).pow(2).pow(0.5).setLabel("SigmaGradient");

        radonData.keySet().stream()
            .sorted()
            .limit(numberOfModels)
            .forEach(county -> createSubModel(radonData.get(county), muBeta, muAlpha, sigmaBeta, sigmaAlpha));

        BayesianNetwork bayesianNetwork = new BayesianNetwork(muAlpha.getConnectedGraph());

        toPlot = Arrays.asList(muAlpha, muBeta, sigmaAlpha, sigmaBeta);

        // note that way too few samples are taken due to time constraints
        probabilisticModel = new KeanuProbabilisticModelWithGradient(bayesianNetwork);
    }

    private void createSubModel(List<Data> data,
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
    }

    public static class Data {
        public String county;
        public double log_radon;
        public double floor;
    }

    @Override
    public NUTS getSamplingAlgorithm() {
        return NUTS.builder()
            .maxTreeHeight(10)
            .saveStatistics(true)
            .build();
    }

    @Override
    public KeanuProbabilisticModelWithGradient getModel() {
        return probabilisticModel;
    }

    @Override
    public List<Vertex> getToPlot() {
        return toPlot;
    }

    @Override
    public int getSampleCount() {
        return 1000;
    }
}
