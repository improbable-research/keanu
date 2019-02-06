package io.improbable.keanu.algorithms.mcmc.nuts;

import io.improbable.keanu.network.KeanuProbabilisticModelWithGradient;
import io.improbable.keanu.vertices.Vertex;

import java.util.List;

public interface ModelForNUTSVis {

    NUTS getSamplingAlgorithm();

    KeanuProbabilisticModelWithGradient getModel();

    List<Vertex> getToPlot();

    int getSampleCount();
}
