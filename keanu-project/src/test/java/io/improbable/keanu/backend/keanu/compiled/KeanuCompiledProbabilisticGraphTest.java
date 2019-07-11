package io.improbable.keanu.backend.keanu.compiled;

import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.probabilistic.PoissonVertex;
import io.improbable.keanu.vertices.intgr.probabilistic.UniformIntVertex;
import org.junit.Test;

import java.util.Map;
import java.util.stream.Collectors;

import static junit.framework.TestCase.assertEquals;

public class KeanuCompiledProbabilisticGraphTest {

    @Test
    public void canMatchGaussian() {
        GaussianVertex A = new GaussianVertex(0, 1);

        matchesLogProb(new BayesianNetwork(A.getConnectedGraph()));
    }

    @Test
    public void canMatchSumOfGaussians() {
        GaussianVertex A = new GaussianVertex(0, 1);
        GaussianVertex B = new GaussianVertex(0, 1);

        DoubleVertex C = A.plus(B);

        matchesLogProb(new BayesianNetwork(C.getConnectedGraph()));
    }

    @Test
    public void canMatchSumOfGaussiansWithObservation() {
        GaussianVertex A = new GaussianVertex(0, 1);
        GaussianVertex B = new GaussianVertex(0, 1);

        GaussianVertex C = new GaussianVertex(A.plus(B), 1);
        C.observe(1);

        matchesLogProb(new BayesianNetwork(C.getConnectedGraph()));
    }

    @Test
    public void canSumUniformIntAndPoissonAndMixWithDoubleObservation() {
        PoissonVertex A = new PoissonVertex(1);
        UniformIntVertex B = new UniformIntVertex(0, 10);

        IntegerVertex C = A.plus(B);
        GaussianVertex observedC = new GaussianVertex(C.toDouble(), 1);
        observedC.observe(2);

        matchesLogProb(new BayesianNetwork(C.getConnectedGraph()));
    }

    private void matchesLogProb(BayesianNetwork bayesianNetwork) {
        KeanuCompiledProbabilisticGraph probabilisticGraph = KeanuCompiledProbabilisticGraph
            .convert(bayesianNetwork);

        Map<VariableReference, Object> inputs = bayesianNetwork.getLatentVertices().stream()
            .collect(Collectors.toMap(Vertex::getReference, Vertex::getValue));

        double logProb = probabilisticGraph.logProb(inputs);

        assertEquals(bayesianNetwork.getLogOfMasterP(), logProb, 0.01);
    }
}
