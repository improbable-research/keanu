package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.algorithms.variational.GradientOptimizer;
import io.improbable.keanu.network.BayesNet;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class SigmoidVertexTest {

    @Test
    public void sigmoidOpIsCalculatedCorrectly() {
        ConstantDoubleVertex x = new ConstantDoubleVertex(3.0);
        SigmoidVertex sigmoidX = new SigmoidVertex(x);

        double expected = 1 / (1 + Math.exp(-3.0));

        assertEquals(expected, sigmoidX.getValue(), 0.0001);
        assertEquals(expected, new SigmoidVertex(3.0).getValue(), 0.0001);
    }

    @Test
    public void sigmoidDualNumberCalculatedCorrectly() {
        DoubleVertex input = new UniformVertex(0, 1);
        input.setAndCascade(0.5);

        DoubleVertex sigmoid = new SigmoidVertex(input);
        double diffSigmoidWrtInput = sigmoid.getDualNumber().getPartialDerivatives().withRespectTo(input);

        double expected = Math.exp(-input.getValue()) / Math.pow(Math.exp(-input.getValue()) + 1, 2);

        assertEquals(expected, diffSigmoidWrtInput, 0.0001);
    }

    @Test
    public void canSolveSigmoidEquation() {
        //sigmoid(x) = 0.75
        //x = -log((1/0.75)-1) = 1.0986

        DoubleVertex unknownX = new UniformVertex(0.0, 10.0);
        unknownX.setAndCascade(5.0);

        SigmoidVertex sigmoid = new SigmoidVertex(unknownX);

        GaussianVertex observableSigmoid = new GaussianVertex(sigmoid, 1.0);
        observableSigmoid.observe(0.75);

        BayesNet bayesNet = new BayesNet(unknownX.getConnectedGraph());
        GradientOptimizer gradientOptimizer = new GradientOptimizer(bayesNet);
        gradientOptimizer.maxLikelihood(5000);

        double mapX = unknownX.getValue();

        assertEquals(1.0986, mapX, 0.001);
    }
}
