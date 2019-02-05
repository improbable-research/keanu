package io.improbable.keanu.algorithms.variational.optimizer.gradient.testcase;

import io.improbable.keanu.algorithms.variational.optimizer.OptimizedResult;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;

import static junit.framework.TestCase.assertTrue;


/**
 * https://en.wikipedia.org/wiki/Himmelblau%27s_function
 * <p>
 * <p>
 * f(x) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
 * <p>
 * Has maxima of 0.0 at four places:
 * (3,2)
 * (-2.805118, 3.131312)
 * (-3.779310, -3.283186)
 * (3.584428, -1.84d81267)
 */
public class HimmelblauTestCase extends BivariateFunctionTestCase {

    public HimmelblauTestCase(double startX, double startY) {
        super(createHimmelblauFunction(startX, startY));
    }

    private static BivariateFunctionTestCase.BivariateFunction createHimmelblauFunction(double startX, double startY) {
        DoubleVertex x = new UniformVertex(-5, 5);
        DoubleVertex y = new UniformVertex(-5, 5);

        x.setValue(startX);
        y.setValue(startY);

        DoubleVertex f1 = x.pow(2).plus(y).minus(11).pow(2);
        DoubleVertex f2 = x.plus(y.pow(2)).minus(7).pow(2);

        DoubleVertex f = f1.plus(f2).times(-1);

        return new BivariateFunctionTestCase.BivariateFunction(x, y, f);
    }

    @Override
    public void assertResult(OptimizedResult result, double actualX, double actualY) {

        double epsilon = 1e-2;

        boolean maximaA = closeTo(new double[]{3.0, 2.0}, actualX, actualY, epsilon);
        boolean maximaB = closeTo(new double[]{-2.805118, 3.131312}, actualX, actualY, epsilon);
        boolean maximaC = closeTo(new double[]{-3.779310, -3.283186}, actualX, actualY, epsilon);
        boolean maximaD = closeTo(new double[]{3.584428, -1.8481267}, actualX, actualY, epsilon);

        boolean matchesOneOfMaxima = maximaA || maximaB || maximaC || maximaD;

        assertTrue(matchesOneOfMaxima);
    }

    private boolean closeTo(double[] expected, double actualX, double actualY, double epsilon) {
        return approxEquals(expected[0], actualX, epsilon) && approxEquals(expected[1], actualY, epsilon);
    }

    private boolean approxEquals(double expected, double actual, double epsilon) {
        return Math.abs(expected - actual) <= Math.abs(epsilon);
    }
}
