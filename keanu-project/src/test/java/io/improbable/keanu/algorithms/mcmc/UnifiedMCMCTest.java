package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.PosteriorSamplingAlgorithm;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.Vertex;
import org.junit.Test;

import static java.util.Collections.emptyList;
import static java.util.Collections.singletonList;
import static org.junit.Assert.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class UnifiedMCMCTest {

    @Test
    public void checkMHIsRunForNonDifferentiableNetwork() {
        BayesianNetwork network = mock(BayesianNetwork.class);
        when(network.getNonDifferentiableVertices()).thenReturn(singletonList(mock(Vertex.class)));

        PosteriorSamplingAlgorithm samplingAlgorithm = MCMC.withDefaultConfig().forNetwork(network);
        assertTrue(samplingAlgorithm instanceof MetropolisHastings);
    }

    @Test
    public void checkNUTSIsRunForDifferentiableNetwork() {
        BayesianNetwork network = mock(BayesianNetwork.class);
        when(network.getNonDifferentiableVertices()).thenReturn(emptyList());

        PosteriorSamplingAlgorithm samplingAlgorithm = MCMC.withDefaultConfig().forNetwork(network);
        assertTrue(samplingAlgorithm instanceof NUTS);
    }

}
