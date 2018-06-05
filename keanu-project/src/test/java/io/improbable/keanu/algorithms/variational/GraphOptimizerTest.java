package io.improbable.keanu.algorithms.variational;

import io.improbable.keanu.network.BayesNetTensorAsContinuous;
import io.improbable.keanu.vertices.dbltensor.DoubleVertex;
import io.improbable.keanu.vertices.dbltensor.probabilistic.GaussianVertex;
import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;

public class GraphOptimizerTest {

    @Test
    public void calculatesMaxLikelihood() {

        DoubleVertex A = new GaussianVertex(20.0, 1.0);
        DoubleVertex B = new GaussianVertex(20.0, 1.0);

        A.setValue(20.0);
        B.setAndCascade(20.0);

        DoubleVertex Cobserved = new GaussianVertex(A.plus(B), 1.0);

        Cobserved.observe(44.0);

        BayesNetTensorAsContinuous bayesNet = new BayesNetTensorAsContinuous(Arrays.asList(A, B, Cobserved));

        TensorGradientOptimizer optimizer = new TensorGradientOptimizer(bayesNet);

        optimizer.maxLikelihood(10000);
        double maxA = A.getValue().scalar();
        double maxB = B.getValue().scalar();

        assertEquals(44, maxA + maxB, 0.1);
    }

    @Test
    public void calculatesMaxAPosteriori() {

        DoubleVertex A = new GaussianVertex(20.0, 1.0);
        DoubleVertex B = new GaussianVertex(20.0, 1.0);

        A.setValue(21.5);
        B.setAndCascade(21.5);

        DoubleVertex Cobserved = new GaussianVertex(A.plus(B), 1.0);

        Cobserved.observe(46.0);

        BayesNetTensorAsContinuous bayesNet = new BayesNetTensorAsContinuous(Arrays.asList(A, B, Cobserved));

        TensorGradientOptimizer optimizer = new TensorGradientOptimizer(bayesNet);

        optimizer.maxAPosteriori(10000);
        double maxA = A.getValue().scalar();
        double maxB = B.getValue().scalar();

        assertEquals(22, maxA, 0.1);
        assertEquals(22, maxB, 0.1);
    }
}
