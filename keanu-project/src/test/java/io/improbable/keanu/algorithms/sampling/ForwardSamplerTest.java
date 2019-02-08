package io.improbable.keanu.algorithms.sampling;

import io.improbable.keanu.Keanu;
import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.TestGraphGenerator;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DoubleBinaryOpVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDouble;
import lombok.extern.slf4j.Slf4j;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertEquals;

@Slf4j
public class ForwardSamplerTest {

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test(expected = IllegalArgumentException.class)
    public void throwIfSampleFromHasUpstreamRandomVariables() {
        GaussianVertex A = new GaussianVertex(100.0, 1);
        DoubleVertex B = A.plus(ConstantVertex.of(5.));
        GaussianVertex C = new GaussianVertex(B, 1.);

        BayesianNetwork network = new BayesianNetwork(A.getConnectedGraph());

        Keanu.Sampling.Forward.withDefaultConfig().sample(network, Collections.singletonList(C), 1000, random);
    }

    @Test(expected = IllegalArgumentException.class)
    public void throwIfSampleFromHasObservations() {
        GaussianVertex A = new GaussianVertex(100.0, 1);
        DoubleVertex B = A.plus(ConstantVertex.of(5.));
        A.observe(100.);

        BayesianNetwork network = new BayesianNetwork(A.getConnectedGraph());

        Keanu.Sampling.Forward.withDefaultConfig().sample(network, Arrays.asList(A, B), 1000, random);
    }

    @Test
    public void canSampleFromPrior() {
        GaussianVertex A = new GaussianVertex(100.0, 1);
        DoubleVertex B = A.plus(ConstantVertex.of(5.));

        BayesianNetwork network = new BayesianNetwork(A.getConnectedGraph());

        final int sampleCount = 1000;
        NetworkSamples samples = Keanu.Sampling.Forward.withDefaultConfig().sample(network, Arrays.asList(A, B), sampleCount, random);

        double averageA = samples.getDoubleTensorSamples(A).getAverages().scalar();
        double averageB = samples.getDoubleTensorSamples(B).getAverages().scalar();

        assertEquals(sampleCount, samples.size());
        assertEquals(100.0, averageA, 0.1);
        assertEquals(105.0, averageB, 0.1);
    }

    @Test
    public void samplesInTopologicalOrder() {
        ConstantDoubleVertex A = ConstantVertex.of(1.0);
        ConstantDoubleVertex B = ConstantVertex.of(2.0);
        ConstantDoubleVertex C = ConstantVertex.of(3.0);

        ArrayList<VertexId> ids = new ArrayList<>();

        IDTrackerVertex trackerOne = new IDTrackerVertex(A, B, ids);
        IDTrackerVertex trackerTwo = new IDTrackerVertex(B, C, ids);
        NonProbabilisticIDTrackerVertex trackerThree = new NonProbabilisticIDTrackerVertex(trackerOne, trackerTwo, ids);
        NonProbabilisticIDTrackerVertex trackerFour = new NonProbabilisticIDTrackerVertex(trackerThree, ConstantVertex.of(5.0), ids);

        trackerOne.setValue(1.);
        trackerTwo.setValue(1.);

        BayesianNetwork network = new BayesianNetwork(A.getConnectedGraph());

        Keanu.Sampling.Forward.withDefaultConfig().sample(network, Arrays.asList(trackerThree, trackerOne, trackerTwo, trackerFour), 1);

        assertEquals(trackerOne.getId(), ids.get(0));
        assertEquals(trackerTwo.getId(), ids.get(1));
        assertEquals(trackerThree.getId(), ids.get(2));
        assertEquals(trackerFour.getId(), ids.get(3));
    }

    @Test
    public void nonProbabilisticVerticesAreRecomputedDuringForwardSample() {
        AtomicInteger opCount = new AtomicInteger(0);

        GaussianVertex A = new GaussianVertex(0, 1);
        TestGraphGenerator.PassThroughVertex B = new TestGraphGenerator.PassThroughVertex(A, opCount, null, id -> log.info("OP on id:" + id));

        BayesianNetwork network = new BayesianNetwork(A.getConnectedGraph());
        Keanu.Sampling.Forward.withDefaultConfig().sample(network, Arrays.asList(A, B), 100, random);

        assertEquals(100, opCount.get());
    }

    public static class IDTrackerVertex extends DoubleVertex implements Differentiable, ProbabilisticDouble {

        private final List<VertexId> ids;

        public IDTrackerVertex(@LoadVertexParam("left")DoubleVertex left, @LoadVertexParam("right")DoubleVertex right, List<VertexId> ids) {
            super(left.getShape());
            this.ids = ids;
            setParents(left, right);
        }

        @Override
        public DoubleTensor sample(KeanuRandom random) {
            ids.add(this.getId());
            return DoubleTensor.ZERO_SCALAR;
        }

        @Override
        public double logProb(DoubleTensor value) {
            return 0;
        }

        @Override
        public Map<Vertex, DoubleTensor> dLogProb(DoubleTensor atValue, Set<? extends Vertex> withRespectTo) {
            return null;
        }
    }

    public static class NonProbabilisticIDTrackerVertex extends DoubleBinaryOpVertex implements Differentiable {

        private final List<VertexId> ids;

        public NonProbabilisticIDTrackerVertex(@LoadVertexParam("left")DoubleVertex left, @LoadVertexParam("right")DoubleVertex right, List<VertexId> ids) {
            super(left, right);
            this.ids = ids;
        }

        @Override
        protected DoubleTensor op(DoubleTensor l, DoubleTensor r) {
            ids.add(this.getId());
            return DoubleTensor.ZERO_SCALAR;
        }
    }

}
