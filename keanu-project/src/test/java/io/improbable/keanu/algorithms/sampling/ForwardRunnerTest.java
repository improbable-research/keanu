package io.improbable.keanu.algorithms.sampling;

import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.junit.Before;
import org.junit.Test;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.backend.KeanuComputableGraph;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.ProbabilisticDouble;

public class ForwardRunnerTest {

    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void samplesFromPrior() {
        GaussianVertex A = new GaussianVertex(100.0, 1);
        GaussianVertex B = new GaussianVertex(A, 1);
        GaussianVertex C = new GaussianVertex(B, 1);
        BayesianNetwork net = new BayesianNetwork(C.getConnectedGraph());
        KeanuComputableGraph graph = new KeanuComputableGraph(C.getConnectedGraph());

        final int sampleCount = 1000;
        NetworkSamples samples = ForwardRunner.sample(graph, net.getLatentVertices(), sampleCount, random);

        double averageC = samples.getDoubleTensorSamples(C).getAverages().scalar();

        assertEquals(sampleCount, samples.size());
        assertEquals(100.0, averageC, 0.1);
    }

    @Test
    public void samplesInTopologicalOrder() {
        ConstantDoubleVertex A = ConstantVertex.of(1.0);
        ConstantDoubleVertex B = ConstantVertex.of(2.0);
        ConstantDoubleVertex C = ConstantVertex.of(3.0);

        ArrayList<VertexId> ids = new ArrayList<>();

        IDTrackerVertex trackerOne = new IDTrackerVertex(A, B, ids);
        IDTrackerVertex trackerTwo = new IDTrackerVertex(B, C, ids);
        IDTrackerVertex trackerThree = new IDTrackerVertex(trackerOne, trackerTwo, ids);

        trackerOne.setValue(1.);
        trackerTwo.setValue(1.);
        trackerThree.setValue(1.);

        BayesianNetwork network = new BayesianNetwork(A.getConnectedGraph());
        KeanuComputableGraph graph = new KeanuComputableGraph(A.getConnectedGraph());

        ForwardRunner.sample(graph, network.getLatentVertices(), 1);

        assertEquals(trackerOne.getId(), ids.get(0));
        assertEquals(trackerTwo.getId(), ids.get(1));
        assertEquals(trackerThree.getId(), ids.get(2));
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

}
