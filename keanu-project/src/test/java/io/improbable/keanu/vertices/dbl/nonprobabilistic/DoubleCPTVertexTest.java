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

    DoubleTensor aValue = DoubleTensor.create(new double[]{0.5, 0.25});
    DoubleTensor bValue = DoubleTensor.create(new double[]{-0.5, -0.25});

    private DoubleCPTVertex doubleCPTNetwork() {
        DoubleVertex A = new GaussianVertex(new int[]{1, 2}, 0, 1);
        A.setValue(aValue);
        DoubleVertex B = new GaussianVertex(0, 1);
        B.observe(bValue);

        BoolVertex leftPredicate = new BernoulliVertex(0.5);
        leftPredicate.observe(true);
        BoolVertex rightPredicate = new BernoulliVertex(0.5);
        rightPredicate.observe(true);

        return ConditionalProbabilityTable.of(leftPredicate, rightPredicate)
            .when(true, true).then(A.times(B))
            .when(true, false).then(A.div(B))
            .when(false, true).then(A.plus(B))
            .orDefault(A.minus(B));
    }

    @Test
    public void canGetFromACondition() {

        DoubleCPTVertex doubleCPTVertex = doubleCPTNetwork();
        DoubleTensor actual = doubleCPTVertex.getValue();
        DoubleTensor expected = aValue.times(bValue);

        assertEquals(expected, actual);
    }

    @Test
    public void canGetDiffFromACondition() {
        DoubleCPTVertex doubleCPTVertex = doubleCPTNetwork();
        BayesianNetwork network = new BayesianNetwork(doubleCPTVertex.getConnectedGraph());

        Vertex<DoubleTensor> A = network.getContinuousLatentVertices().get(0);

        DoubleTensor actual = doubleCPTVertex.getDualNumber().getPartialDerivatives().withRespectTo(A);
        DoubleTensor expected = bValue.diag().reshape(1, 2, 1, 2);

        assertEquals(expected, actual);
    }

}
