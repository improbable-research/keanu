package io.improbable.keanu.vertices.generic.nonprobabilistic;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.sampling.Prior;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.probabilistic.Flip;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import org.junit.Before;
import org.junit.Test;

import java.util.Collections;

import static io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBoolVertex.FALSE;
import static io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBoolVertex.TRUE;
import static junit.framework.TestCase.assertFalse;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class IfVertexTest {

    private KeanuRandom random;
    private Flip v1;
    private Flip v2;
    private double pV2 = 0.4;
    private double pV1 = 0.25;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
        v1 = new Flip(pV1);
        v2 = new Flip(pV2);
    }

    @Test
    public void functionsAsIf() {

        BoolVertex predicate = new Flip(0.5);

        BoolVertex ifIsTrue = If.isTrue(predicate)
            .then(TRUE)
            .orElse(FALSE);

        predicate.setAndCascade(true);
        assertTrue(ifIsTrue.getValue().scalar());

        predicate.setAndCascade(false);
        assertFalse(ifIsTrue.getValue().scalar());
    }

    @Test
    public void ifProbabilityIsCorrect() {

        double pV3 = 0.1;
        Flip v3 = new Flip(pV3);

        BoolVertex v4 = If.isTrue(v1)
            .then(v2)
            .orElse(v3);

        double pV4True = ifProbability(pV1, pV2, pV3);

        assertEquals(priorProbabilityTrue(v4, 10000, random), pV4True, 0.01);
    }

    private double ifProbability(double pThn, double pThnIsValue, double pElsIsValue) {
        double pThnAndThnIsValue = pThn * pThnIsValue;
        double pElsAndElsIsValue = (1 - pThn) * pElsIsValue;

        return pThnAndThnIsValue + pElsAndElsIsValue;
    }

    public static double priorProbabilityTrue(Vertex<? extends Tensor<Boolean>> vertex, int sampleCount, KeanuRandom random) {
        BayesianNetwork net = new BayesianNetwork(vertex.getConnectedGraph());

        NetworkSamples samples = Prior.sample(net, Collections.singletonList(vertex), sampleCount, random);
        return samples.get(vertex).probability(val -> val.scalar());
    }
}
