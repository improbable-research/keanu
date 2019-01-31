package io.improbable.keanu.algorithms.variational.optimizer.gradient.testcase;

import io.improbable.keanu.algorithms.variational.optimizer.OptimizedResult;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;

import static junit.framework.TestCase.assertEquals;

/**
 * This is a common test case for gradient optimization algorithms.
 * <p>
 * https://en.wikipedia.org/wiki/Rosenbrock_function
 * <p>
 * f(x,y) = (a - x)^2 + b*(y - x^2)^2
 * <p>
 * The function has been inverted here because our optimizers find maxima and this
 * function provides a global minima. The global maxima is at (a, a^2) so for a=1,b=100
 * the maxima will be (1,1)
 */
public class RosenbrockTestCase extends BivariateFunctionTestCase {

    private final double expectedX, expectedY;

    public RosenbrockTestCase(double aParameter, double bParameter) {
        super(createRosenbrockFunction(aParameter, bParameter));

        this.expectedX = aParameter;
        this.expectedY = aParameter * aParameter;

    }

    private static BivariateFunctionTestCase.BivariateFunction createRosenbrockFunction(double aParameter, double bParameter) {
        DoubleVertex a = new ConstantDoubleVertex(aParameter);
        DoubleVertex b = new ConstantDoubleVertex(bParameter);

        DoubleVertex x = new UniformVertex(-1000, 1000);
        DoubleVertex y = new UniformVertex(-1000, 1000);

        x.setValue(0);
        y.setValue(3);

        DoubleVertex f = a.minus(x).pow(2).plus(
            b.times(y.minus(x.pow(2)).pow(2))
        ).times(-1);

        return new BivariateFunction(x, y, f);
    }

    @Override
    public void assertResult(OptimizedResult result, double actualX, double actualY) {
        assertEquals(expectedX, actualX, 1e-2);
        assertEquals(expectedY, actualY, 1e-2);
    }
}
