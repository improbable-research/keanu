package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.Keanu;
import io.improbable.keanu.algorithms.PosteriorSamplingAlgorithm;
import io.improbable.keanu.algorithms.mcmc.nuts.NUTS;
import io.improbable.keanu.algorithms.variational.optimizer.KeanuProbabilisticModel;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.FloorVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

public class UnifiedMCMCTest {

    @Test
    public void checkMHIsRunForNonDifferentiableNetwork() {
        GaussianVertex gaussianA = new GaussianVertex(5., 1.);
        FloorVertex nonDiffable =  new FloorVertex(gaussianA);
        GaussianVertex postNonDiffLatent = new GaussianVertex(nonDiffable, 1.);
        KeanuProbabilisticModel model = new KeanuProbabilisticModel(postNonDiffLatent.getConnectedGraph());

        PosteriorSamplingAlgorithm samplingAlgorithm = Keanu.Sampling.MCMC.withDefaultConfigFor(model);
        assertTrue(samplingAlgorithm instanceof MetropolisHastings);
    }

    @Test
    public void checkNUTSIsRunForDifferentiableNetwork() {
        GaussianVertex gaussianA = new GaussianVertex(5., 1.);
        GaussianVertex gaussianB = new GaussianVertex(gaussianA, 1.);
        KeanuProbabilisticModel model = new KeanuProbabilisticModel(gaussianB.getConnectedGraph());

        PosteriorSamplingAlgorithm samplingAlgorithm = Keanu.Sampling.MCMC.withDefaultConfigFor(model);
        assertTrue(samplingAlgorithm instanceof NUTS);
    }

}
