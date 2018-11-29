package io.improbable.keanu.vertices.dbl.nonprobabilistic;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.ConditionalProbabilityTable;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class DoubleCPTVertexTest {

    private DoubleTensor aValue = DoubleTensor.create(0.5, 0.25);
    private DoubleTensor bValue = DoubleTensor.create(-0.5, -0.25);

    private DoubleCPTVertex doubleCPTNetwork(boolean left, boolean right) {
        DoubleVertex A = new GaussianVertex(new long[]{2}, 0, 1);
        A.setValue(aValue);
        DoubleVertex B = new GaussianVertex(0, 1);
        B.observe(bValue);

        BoolVertex leftPredicate = new BernoulliVertex(0.5);
        leftPredicate.observe(left);
        BoolVertex rightPredicate = new BernoulliVertex(0.5);
        rightPredicate.observe(right);

        return ConditionalProbabilityTable.of(leftPredicate, rightPredicate)
            .when(true, true).then(A.times(B))
            .when(true, false).then(A.div(B))
            .when(false, true).then(A.plus(B))
            .orDefault(B.minus(A));
    }

    @Test
    public void canGetFromACondition() {
        assertFromACondition(true, true, aValue.times(bValue));
        assertFromACondition(true, false, aValue.div(bValue));
        assertFromACondition(false, true, aValue.plus(bValue));
        assertFromACondition(false, false, bValue.minus(aValue));
    }

    private void assertFromACondition(boolean left, boolean right, DoubleTensor expected) {
        DoubleCPTVertex doubleCPTVertex = doubleCPTNetwork(left, right);
        DoubleTensor actual = doubleCPTVertex.getValue();
        assertEquals(expected, actual);
    }

    @Test
    public void canGetDiffFromACondition() {
        long[] expectedShape = new long[]{2, 2};
        assertDiffFromACondition(true, true, bValue.diag().reshape(expectedShape));
        assertDiffFromACondition(true, false, bValue.reciprocal().diag().reshape(expectedShape));
        assertDiffFromACondition(false, true, DoubleTensor.eye(2).reshape(expectedShape));
        assertDiffFromACondition(false, false, DoubleTensor.eye(2).reshape(expectedShape).unaryMinus());
    }

    private void assertDiffFromACondition(boolean left, boolean right, DoubleTensor expected) {
        DoubleCPTVertex doubleCPTVertex = doubleCPTNetwork(left, right);
        BayesianNetwork network = new BayesianNetwork(doubleCPTVertex.getConnectedGraph());

        Vertex<DoubleTensor> A = network.getContinuousLatentVertices().get(0);

        DoubleTensor actual = doubleCPTVertex.getDerivativeWrtLatents().withRespectTo(A);

        assertEquals(expected, actual);
    }

}
