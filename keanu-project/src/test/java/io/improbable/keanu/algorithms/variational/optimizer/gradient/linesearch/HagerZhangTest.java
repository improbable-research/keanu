package io.improbable.keanu.algorithms.variational.optimizer.gradient.linesearch;

import io.improbable.keanu.algorithms.variational.optimizer.gradient.ApacheFitnessFunctionAdapter;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.ApacheFitnessFunctionGradientAdapter;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.testcase.BivariateFunctionTestCase;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.UniformVertex;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;

public class HagerZhangTest {

    @Test
    public void findsLowerFitnessInDirection() {

        DoubleVertex x = new UniformVertex(-1000, 1000);
        DoubleVertex y = new UniformVertex(-1000, 1000);
        DoubleVertex f = x.pow(2).plus(y.pow(2)).unaryMinus();

        BivariateFunctionTestCase.BivariateFunction bivariateFunction = new BivariateFunctionTestCase.BivariateFunction(x, y, f);

        ApacheFitnessFunctionAdapter fitness = new ApacheFitnessFunctionAdapter(
            bivariateFunction.getFitnessFunction(), bivariateFunction.getVariables()
        );

        ApacheFitnessFunctionGradientAdapter gradient = new ApacheFitnessFunctionGradientAdapter(
            bivariateFunction.getFitnessFunctionGradient(),
            bivariateFunction.getVariables()
        );

        DoubleTensor position = DoubleTensor.create(1, 1);

        DoubleTensor g = DoubleTensor.create(gradient.value(position.asFlatDoubleArray())).unaryMinus();

        HagerZhang hagerZhang = HagerZhang.builder().build();
        DoubleTensor searchDirection = g.unaryMinus();

        HagerZhang.Results results = hagerZhang.lineSearch(position, searchDirection, fitness, gradient, 1.0);

        DoubleTensor resultPosition = position.plus(searchDirection.times(results.alpha));

        assertArrayEquals(new double[]{0, 0}, resultPosition.asFlatDoubleArray(), 1e-3);
    }
}
