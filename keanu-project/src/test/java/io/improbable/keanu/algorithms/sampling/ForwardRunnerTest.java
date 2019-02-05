package io.improbable.keanu.algorithms.sampling;

import static org.junit.Assert.assertEquals;

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
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
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

    }

    public static class IncrementVertex extends DoubleVertex implements Differentiable, ProbabilisticDouble {

        private final DoubleVertex left;
        private final DoubleVertex right;
        private final List<VertexId> ids;

        public IncrementVertex(@LoadVertexParam("left")DoubleVertex left, @LoadVertexParam("right")DoubleVertex right, List<VertexId> ids) {
            super(left.getShape());
            this.left = left;
            this.right = right;
            this.ids = ids;
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
