package io.improbable.keanu.vertices.generic.nonprobabilistic;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.bool.BooleanVertexTest;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.tensor.If;
import org.junit.Before;
import org.junit.Test;

import static io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBooleanVertex.FALSE;
import static io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBooleanVertex.TRUE;
import static junit.framework.TestCase.assertFalse;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class IfVertexTest {

    private KeanuRandom random;
    private BernoulliVertex v1;
    private BernoulliVertex v2;
    private double pV2 = 0.4;
    private double pV1 = 0.25;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
        v1 = new BernoulliVertex(pV1);
        v2 = new BernoulliVertex(pV2);
    }

    @Test
    public void functionsAsIf() {

        BooleanVertex predicate = new BernoulliVertex(0.5);

        BooleanVertex ifIsTrue = If.isTrue(predicate)
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
        BernoulliVertex v3 = new BernoulliVertex(pV3);

        BooleanVertex v4 = If.isTrue(v1)
            .then(v2)
            .orElse(v3);

        double pV4True = ifProbability(pV1, pV2, pV3);

        assertEquals(BooleanVertexTest.priorProbabilityTrue(v4, 10000, random), pV4True, 0.01);
    }

    private double ifProbability(double pThn, double pThnIsValue, double pElsIsValue) {
        double pThnAndThnIsValue = pThn * pThnIsValue;
        double pElsAndElsIsValue = (1 - pThn) * pElsIsValue;

        return pThnAndThnIsValue + pElsAndElsIsValue;
    }

}
