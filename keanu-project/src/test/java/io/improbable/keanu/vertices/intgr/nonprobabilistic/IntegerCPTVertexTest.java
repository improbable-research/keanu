package io.improbable.keanu.vertices.intgr.nonprobabilistic;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.ConditionalProbabilityTable;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class IntegerCPTVertexTest {

    private IntegerTensor aValue = IntegerTensor.create(3, 4);
    private IntegerTensor bValue = IntegerTensor.create(5, 6);

    private IntegerCPTVertex integerCPTNetwork(boolean left, boolean right) {
        IntegerVertex A = new ConstantIntegerVertex(aValue);
        IntegerVertex B = new ConstantIntegerVertex(bValue);

        BooleanVertex leftPredicate = new BernoulliVertex(0.5);
        leftPredicate.observe(left);
        BooleanVertex rightPredicate = new BernoulliVertex(0.5);
        rightPredicate.observe(right);

        return ConditionalProbabilityTable.of(leftPredicate, rightPredicate)
            .when(true, true).then(A.times(B))
            .when(true, false).then(A.div(B))
            .when(false, true).then(A.plus(B))
            .orDefault(B.minus(A));
    }

    private void assertFromACondition(boolean left, boolean right, IntegerTensor expected) {
        IntegerCPTVertex integerCPTVertex = integerCPTNetwork(left, right);
        IntegerTensor actual = integerCPTVertex.getValue();
        assertEquals(expected, actual);
    }

    @Test
    public void canGetFromACondition() {
        assertFromACondition(true, true, aValue.times(bValue));
        assertFromACondition(true, false, aValue.div(bValue));
        assertFromACondition(false, true, aValue.plus(bValue));
        assertFromACondition(false, false, bValue.minus(aValue));
    }
}
