package io.improbable.keanu.algorithms.variational;

import io.improbable.keanu.network.BayesNet;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;

public class GraphOptimizerTest {

    @Test
    public void calculatesMaxLikelihood() {

        DoubleVertex A = new GaussianVertex(new ConstantDoubleVertex(20.0), new ConstantDoubleVertex(1.0));
        DoubleVertex B = new GaussianVertex(new ConstantDoubleVertex(20.0), new ConstantDoubleVertex(1.0));

        A.setValue(20.0);
        B.setAndCascade(20.0);

        DoubleVertex Cobserved = new GaussianVertex(A.plus(B), new ConstantDoubleVertex(1.0));

        Cobserved.observe(44.0);

        BayesNet bayesNet = new BayesNet(Arrays.asList(A, B, Cobserved));

        GradientOptimizer optimizer = new GradientOptimizer(bayesNet);

        optimizer.maxLikelihood(10000);
        double maxA = A.getValue();
        double maxB = B.getValue();

        assertEquals(44, maxA + maxB, 0.1);
    }

    @Test
    public void calculatesMaxAPosteriori() {

        DoubleVertex A = new GaussianVertex(new ConstantDoubleVertex(20.0), new ConstantDoubleVertex(1.0));
        DoubleVertex B = new GaussianVertex(new ConstantDoubleVertex(20.0), new ConstantDoubleVertex(1.0));

        A.setValue(21.5);
        B.setAndCascade(21.5);

        DoubleVertex Cobserved = new GaussianVertex(A.plus(B), new ConstantDoubleVertex(1.0));

        Cobserved.observe(46.0);

        BayesNet bayesNet = new BayesNet(Arrays.asList(A, B, Cobserved));

        GradientOptimizer optimizer = new GradientOptimizer(bayesNet);

        optimizer.maxAPosteriori(10000);
        double maxA = A.getValue();
        double maxB = B.getValue();

        assertEquals(22, maxA, 0.1);
        assertEquals(22, maxB, 0.1);
    }
}
