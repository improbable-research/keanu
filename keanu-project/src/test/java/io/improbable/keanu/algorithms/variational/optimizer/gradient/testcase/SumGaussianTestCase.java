package io.improbable.keanu.algorithms.variational.optimizer.gradient.testcase;

import io.improbable.keanu.algorithms.variational.optimizer.OptimizerTestCase;
import io.improbable.keanu.algorithms.variational.optimizer.OptimizedResult;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;

public class SumGaussianTestCase implements OptimizerTestCase {

    private final DoubleVertex A;
    private final DoubleVertex B;
    private final BayesianNetwork model;

    public SumGaussianTestCase() {
        A = new GaussianVertex(20.0, 1.0);
        B = new GaussianVertex(20.0, 1.0);

        A.setValue(20.0);
        B.setAndCascade(20.0);

        DoubleVertex Cobserved = new GaussianVertex(A.plus(B), 1.0);

        Cobserved.observe(46.0);

        model = new BayesianNetwork(Arrays.asList(A, B, Cobserved));
    }

    @Override
    public BayesianNetwork getModel() {
        return model;
    }

    @Override
    public void assertMLE(OptimizedResult result) {

        double maxA = result.get(A.getReference()).scalar();
        double maxB = result.get(B.getReference()).scalar();

        assertEquals(46, maxA + maxB, 0.1);
    }

    @Override
    public void assertMAP(OptimizedResult result) {
        double maxA = result.get(A.getReference()).scalar();
        double maxB = result.get(B.getReference()).scalar();

        assertEquals(22, maxA, 0.1);
        assertEquals(22, maxB, 0.1);
    }

}
