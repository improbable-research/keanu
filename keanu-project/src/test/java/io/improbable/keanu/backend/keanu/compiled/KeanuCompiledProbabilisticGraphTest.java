package io.improbable.keanu.backend.keanu.compiled;

import io.improbable.keanu.backend.VariableReference;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.AdditionVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
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

        AdditionVertex C = A.plus(B);

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

    private void matchesLogProb(BayesianNetwork bayesianNetwork) {
        KeanuCompiledProbabilisticGraph probabilisticGraph = KeanuCompiledProbabilisticGraph
            .convert(bayesianNetwork);

        Map<VariableReference, Object> inputs = bayesianNetwork.getLatentVertices().stream()
            .collect(Collectors.toMap(Vertex::getReference, Vertex::getValue));

        double logProb = probabilisticGraph.logProb(inputs);

        assertEquals(bayesianNetwork.getLogOfMasterP(), logProb, 0.01);
    }
}
