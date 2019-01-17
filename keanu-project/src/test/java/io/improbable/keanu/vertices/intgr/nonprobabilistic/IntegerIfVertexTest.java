package io.improbable.keanu.vertices.intgr.nonprobabilistic;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.generic.nonprobabilistic.If;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertexTest;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class IntegerIfVertexTest {

    @Rule
    public DeterministicRule deterministicRule = new DeterministicRule();
    private KeanuRandom random;

    @Before
    public void setup() {
        random = new KeanuRandom(1);
    }

    @Test
    public void correctBranchTaken() {

        BooleanVertex predicate = new BernoulliVertex(0.5);

        IntegerVertex ifIsTrue = If.isTrue(predicate)
            .then(1)
            .orElse(0);

        predicate.setAndCascade(true);
        assertEquals(1, (int) ifIsTrue.getValue().scalar());

        predicate.setAndCascade(false);
        assertEquals(0, (int) ifIsTrue.getValue().scalar());
    }

    @Test
    public void expectedValueMatchesBranchProbability() {

        double p = 0.5;
        int thenValue = 2;
        int elseValue = 4;
        double expectedMean = p * thenValue + (1 - p) * elseValue;

        BernoulliVertex predicate = new BernoulliVertex(p);

        IntegerVertex vertex = If.isTrue(predicate)
            .then(thenValue)
            .orElse(elseValue);

        assertEquals(IntegerVertexTest.calculateMeanOfVertex(vertex), expectedMean, 0.1);
    }

}
