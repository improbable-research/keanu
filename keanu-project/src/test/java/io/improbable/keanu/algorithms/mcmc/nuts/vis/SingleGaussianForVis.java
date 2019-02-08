package io.improbable.keanu.algorithms.mcmc.nuts.vis;

import io.improbable.keanu.Keanu;
import io.improbable.keanu.algorithms.mcmc.nuts.NUTS;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.KeanuProbabilisticModelWithGradient;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import lombok.Getter;

import java.util.Arrays;
import java.util.List;

public class SingleGaussianForVis implements ModelForNUTSVis {

    @Getter
    private NUTS samplingAlgorithm;

    @Getter
    private KeanuProbabilisticModelWithGradient model;

    @Getter
    private List<Vertex> toPlot;

    @Getter
    private int sampleCount;

    public SingleGaussianForVis() {

        GaussianVertex A = new GaussianVertex(0, 1);
//        A.setValue(3.0);

        model = new KeanuProbabilisticModelWithGradient(new BayesianNetwork(A.getConnectedGraph()));

        sampleCount = 3000;

        samplingAlgorithm = Keanu.Sampling.NUTS.builder()
            .adaptCount(sampleCount)
            .saveStatistics(true)
            .build();

        toPlot = Arrays.asList(A);
    }
}
