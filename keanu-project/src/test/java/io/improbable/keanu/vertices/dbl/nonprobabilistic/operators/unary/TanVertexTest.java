package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.algorithms.variational.GradientOptimizer;
import io.improbable.keanu.network.BayesNet;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
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
        //tan 3x = -1
        //x = 45, 105 or 165

        DoubleVertex unknownTheta = new UniformVertex(0.0, 10.0);
        unknownTheta.setValue(5.0);

        TanVertex tan = new TanVertex(unknownTheta.multiply(3.0));

        GaussianVertex observableTan = new GaussianVertex(tan, 1.0);
        observableTan.observe(-1.0);

        BayesNet bayesNet = new BayesNet(unknownTheta.getConnectedGraph());
        GradientOptimizer gradientOptimizer = new GradientOptimizer(bayesNet);

        gradientOptimizer.maxLikelihood(5000);

        double result = Math.tan(3 * unknownTheta.getValue());

        assertEquals(-1, result, 0.001);
    }

    @Test
    public void canSolveTanIdentity() {
        //Tan(PI / 2 - X) = cos(x) / sin(x)

        List<Double> data = new ArrayList<>();
        int dataCount = 100;

        for (int i = 1; i < dataCount; i++) {
            data.add(Math.tan(Math.PI / 2 - i));
        }

        DoubleVertex unknownConstant = new UniformVertex(0.0, 5.0);
        unknownConstant.setValue(2.5);

        for (int j = 1; j < dataCount; j++) {
            DoubleVertex tanPiOver2MinusX = new TanVertex(unknownConstant.minus(j));
            DoubleVertex CosOverSin = new CosVertex(j).div(new SinVertex(j));

            GaussianVertex observableTan = new GaussianVertex(tanPiOver2MinusX, .00001);
            GaussianVertex observableCosOverSin = new GaussianVertex(CosOverSin, .00001);

            observableTan.observe(data.get(j - 1));
            observableCosOverSin.observe(data.get(j - 1));
        }

        BayesNet bayesNet = new BayesNet(unknownConstant.getConnectedGraph());
        GradientOptimizer gradientOptimizer = new GradientOptimizer(bayesNet);

        gradientOptimizer.maxLikelihood(1500);

        assertEquals(Math.PI / 2, Math.abs(unknownConstant.getValue()) % Math.PI, 0.001);
    }

    @Test
    public void tanDualNumberIsCalculatedCorrectly() {
        UniformVertex uniformVertex = new UniformVertex(0, 10);
        uniformVertex.setValue(5.0);

        TanVertex tan = new TanVertex(uniformVertex);

        double dTan = tan.getDualNumber().getPartialDerivatives().withRespectTo(uniformVertex);
        //dTan = 1 / sec^2(x)
        double expected = 1 / (Math.pow(Math.cos(5.0), 2));

        assertEquals(expected, dTan, 0.001);
    }

}
