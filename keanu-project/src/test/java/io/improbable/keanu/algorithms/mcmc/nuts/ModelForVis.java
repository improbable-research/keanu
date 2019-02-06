package io.improbable.keanu.algorithms.mcmc.nuts;

import io.improbable.keanu.Keanu;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.KeanuProbabilisticModelWithGradient;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.AdditionVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import lombok.Getter;

import java.util.Arrays;
import java.util.List;

public class ModelForVis implements ModelForNUTSVis {

    @Getter
    private NUTS samplingAlgorithm;

    @Getter
    private KeanuProbabilisticModelWithGradient model;

    @Getter
    private List<Vertex> toPlot;

    @Getter
    private int sampleCount;

    public ModelForVis() {

        GaussianVertex A = new GaussianVertex(20, 1);
        A.setLabel("A");
//        A.setValue(-10);
        GaussianVertex B = new GaussianVertex(20, 1);
        B.setLabel("B");
//        B.setValue(-10);
        AdditionVertex D = A.plus(B);
        GaussianVertex C = new GaussianVertex(D, 1);
        C.observe(46);

        model = new KeanuProbabilisticModelWithGradient(new BayesianNetwork(A.getConnectedGraph()));
//        KeanuOptimizer.Gradient.builderFor(A.getConnectedGraph()).build().maxAPosteriori();

        sampleCount = 6000;

        samplingAlgorithm = Keanu.Sampling.NUTS.builder()
            .adaptCount(sampleCount)
            .maxTreeHeight(7)
//            .adaptEnabled(false)
            .saveStatistics(true)
            .build();

        toPlot = Arrays.asList(A, B);
    }
}
