package io.improbable.keanu.algorithms.variational;

import static org.junit.Assert.assertEquals;

import java.util.Arrays;

import org.junit.Test;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.VertexOfType;

public class GraphOptimizerTest {

    @Test
    public void calculatesMaxLikelihood() {

        DoubleVertex A = VertexOfType.gaussian(20.0, 1.0);
        DoubleVertex B = VertexOfType.gaussian(20.0, 1.0);

        A.setValue(20.0);
        B.setAndCascade(20.0);

        DoubleVertex Cobserved = VertexOfType.gaussian(A.plus(B), ConstantVertex.of(1.0));

        Cobserved.observe(DoubleTensor.scalar(44.0));

        BayesianNetwork bayesNet = new BayesianNetwork(Arrays.asList(A, B, Cobserved));

        GradientOptimizer optimizer = new GradientOptimizer(bayesNet);

        optimizer.maxLikelihood();
        double maxA = A.getValue().scalar();
        double maxB = B.getValue().scalar();

        assertEquals(44, maxA + maxB, 0.1);
    }

    @Test
    public void calculatesMaxAPosteriori() {

        DoubleVertex A = VertexOfType.gaussian(20.0, 1.0);
        DoubleVertex B = VertexOfType.gaussian(20.0, 1.0);

        A.setValue(21.5);
        B.setAndCascade(21.5);

        DoubleVertex Cobserved = VertexOfType.gaussian(A.plus(B), ConstantVertex.of(1.0));

        Cobserved.observe(DoubleTensor.scalar(46.0));

        BayesianNetwork bayesNet = new BayesianNetwork(Arrays.asList(A, B, Cobserved));

        GradientOptimizer optimizer = new GradientOptimizer(bayesNet);

        optimizer.maxAPosteriori();
        double maxA = A.getValue().scalar();
        double maxB = B.getValue().scalar();

        assertEquals(22, maxA, 0.1);
        assertEquals(22, maxB, 0.1);
    }
}
