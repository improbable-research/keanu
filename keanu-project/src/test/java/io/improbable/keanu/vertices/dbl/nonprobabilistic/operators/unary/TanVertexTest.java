package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.algorithms.variational.GradientOptimizer;
import io.improbable.keanu.distributions.continuous.Gaussian;
import io.improbable.keanu.network.BayesNet;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Assert;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class TanVertexTest {


    @Test
    public void tanOpIsCalculatedCorrectly() {
        ConstantDoubleVertex x = new ConstantDoubleVertex(Math.PI / 2);
        TanVertex tan = new TanVertex(x);

        assertEquals(Math.tan(Math.PI / 2), tan.getValue(), 0.0001);
    }

    @Test
    public void tanOpIsCalculatedCorrectlyWithValue() {
        TanVertex tan = new TanVertex(Math.PI / 2);

        assertEquals(Math.tan(Math.PI / 2), tan.getValue(), 0.0001);
    }

    @Test
    public void canSolveTanEquation() {
        //tan 3x = 1
        //x = 45, 105 or 165

        DoubleVertex unknownTheta = new UniformVertex(0.0, 10.0);
        TanVertex tan = new TanVertex(unknownTheta.multiply(3.0));

        GaussianVertex observableTan = new GaussianVertex(tan, 1.0);
        observableTan.observe(-1.0);

        BayesNet bayesNet = new BayesNet(unknownTheta.getConnectedGraph());
        GradientOptimizer gradientOptimizer = new GradientOptimizer(bayesNet);

        gradientOptimizer.maxLikelihood(5000);

        double theta = Math.toDegrees(unknownTheta.getValue()) % 60;

        assertEquals(45, theta, 0.001);
    }

}
