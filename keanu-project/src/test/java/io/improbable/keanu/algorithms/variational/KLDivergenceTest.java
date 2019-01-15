package io.improbable.keanu.algorithms.variational;

import com.google.common.collect.Iterables;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.algorithms.variational.optimizer.KeanuProbabilisticModel;
import io.improbable.keanu.algorithms.variational.optimizer.ProbabilisticModel;
import io.improbable.keanu.algorithms.variational.optimizer.VariableReference;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.NetworkState;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDouble;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.lessThan;
import static org.junit.Assert.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class KLDivergenceTest {

    @Rule
    public ExpectedException thrown = ExpectedException.none();

    private QDistribution qDist;

    @Before
    public void setup() {
        qDist = mock(QDistribution.class);
    }

    @Test
    public void klDivergenceIsZeroIfPAndQAreIdentical() {
        double identicalProb = Math.log(0.5);

        when(qDist.getLogOfMasterP(any(NetworkState.class))).thenReturn(identicalProb);
        NetworkSamples samples = createNetworkSamplesWithOneVertexAndOneSample(identicalProb);

        assertEquals(0., KLDivergence.compute(qDist, samples), 1e-6);
    }

    @Test
    public void throwsExceptionIfPProbIsNotZeroButQIs() {
        when(qDist.getLogOfMasterP(any(NetworkState.class))).thenReturn(Double.NEGATIVE_INFINITY);
        NetworkSamples samples = createNetworkSamplesWithOneVertexAndOneSample(Math.log(0.5));

        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("Q cannot have smaller support than P");

        KLDivergence.compute(qDist, samples);
    }

    @Test
    public void klDivergenceReturnsZeroIfPIsZeroButQIsNot() {
        when(qDist.getLogOfMasterP(any(NetworkState.class))).thenReturn(Math.log(0.5));
        NetworkSamples samples = createNetworkSamplesWithOneVertexAndOneSample(Double.NEGATIVE_INFINITY);

        assertEquals(0., KLDivergence.compute(qDist, samples), 1e-10);
    }

    @Test
    public void klDivergenceReturnsZeroIfPAndQAreZero() {
        when(qDist.getLogOfMasterP(any(NetworkState.class))).thenReturn(Double.NEGATIVE_INFINITY);
        NetworkSamples samples = createNetworkSamplesWithOneVertexAndOneSample(Double.NEGATIVE_INFINITY);

        assertEquals(0., KLDivergence.compute(qDist, samples), 1e-10);
    }

    @Test
    public void returnsLargerKLDivergenceIfTheLocationOfQIsFurtherFromP_QIsProbabilisticDouble() {
        GaussianVertex v1 = new GaussianVertex(0., 1.);
        ConstantDoubleVertex v2 = new ConstantDoubleVertex(0.1);
        DoubleVertex v3 = v1.plus(v2);

        ProbabilisticModel network = new KeanuProbabilisticModel(new BayesianNetwork(v3.getConnectedGraph()));
        NetworkSamples samples = MetropolisHastings
            .withDefaultConfig()
            .getPosteriorSamples(network, Collections.singletonList(v1), 1000);

        ProbabilisticDouble q1 = new GaussianVertex(0.1, 1.);
        ProbabilisticDouble q2 = new GaussianVertex(10.0, 1.);

        assertThat(KLDivergence.compute(q1, samples), lessThan(KLDivergence.compute(q2, samples)));
    }

    @Test
    public void returnsLargerKLDivergenceIfTheLocationOfQIsFurtherFromP_QIsQDistribution() {
        GaussianVertex v1 = new GaussianVertex(0., 1.);
        ConstantDoubleVertex v2 = new ConstantDoubleVertex(0.1);
        DoubleVertex v3 = v1.plus(v2);

        ProbabilisticModel network = new KeanuProbabilisticModel(new BayesianNetwork(v3.getConnectedGraph()));
        NetworkSamples samples = MetropolisHastings
            .withDefaultConfig()
            .getPosteriorSamples(network, Collections.singletonList(v1), 1000);

        QDistribution q1 = new TestGaussianQDistribution(0.1, 1.);
        QDistribution q2 = new TestGaussianQDistribution(10.0, 1.);

        assertThat(KLDivergence.compute(q1, samples), lessThan(KLDivergence.compute(q2, samples)));
    }

    @Test
    public void throwsExceptionIfNetworkStateHasMoreThanOneVertexAndQIsProbabilisticDouble() {
        GaussianVertex v1 = new GaussianVertex(0., 1.);
        ConstantDoubleVertex v2 = new ConstantDoubleVertex(0.1);
        DoubleVertex v3 = v1.plus(v2);

        ProbabilisticModel network = new KeanuProbabilisticModel(new BayesianNetwork(v3.getConnectedGraph()));
        NetworkSamples samples = MetropolisHastings
            .withDefaultConfig()
            .getPosteriorSamples(network, Arrays.asList(v1, v3), 1000);
        ProbabilisticDouble q = new GaussianVertex(0.1, 1.);

        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("A NetworkState does not contain exactly 1 vertex and ProbabilisticDouble can only compute the log probability of one value. Try computing KL divergence against a QDistribution instead.");

        KLDivergence.compute(q, samples);
    }

    private NetworkSamples createNetworkSamplesWithOneVertexAndOneSample(double p) {
        Map<VariableReference, List<DoubleTensor>> samplesByVertex = new HashMap<>();
        samplesByVertex.put(new VertexId(), Collections.singletonList(DoubleTensor.scalar(1.)));
        List<Double> logOfMasterPForEachSample = Collections.singletonList(p);

        return new NetworkSamples(samplesByVertex, logOfMasterPForEachSample, 1);
    }

    private class TestGaussianQDistribution extends GaussianVertex implements QDistribution {

        TestGaussianQDistribution(double mu, double sigma) {
            super(mu, sigma);
        }

        @Override
        public double getLogOfMasterP(NetworkState state) {
            DoubleTensor vertexValue = state.get(Iterables.getOnlyElement(state.getVariableReferences()));
            return logPdf(vertexValue);
        }
    }

}