package io.improbable.keanu.algorithms.mcmc;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.algorithms.PosteriorSamplingAlgorithm;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Test;

import static java.util.Collections.emptyList;
import static java.util.Collections.singletonList;
import static org.junit.Assert.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class UnifiedMCMCTest {

    @Test
    public void checksIfNetworkIsDifferentiable() {
        BayesianNetwork network = mock(BayesianNetwork.class);
        Vertex vertex = new GaussianVertex(0, 1);
        when(network.getTopLevelLatentVertices()).thenReturn(ImmutableList.of(vertex));
        when(network.getLatentOrObservedVertices()).thenReturn(ImmutableList.of(vertex));
        when(network.getContinuousLatentVertices()).thenReturn(ImmutableList.of(vertex));

        MCMC.withDefaultConfig().forNetwork(network);
        verify(network).getNonDifferentiableVertices();
    }

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
